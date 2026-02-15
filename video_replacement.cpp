// Allows video replacement of FMVS (STR) with the same filename as a external MP4 file, loaded from resources/videos/FILENAME.MP4.
//Filenames must match the STR name in order for it to play.
//Encoding doesn't matter. If it's an Mp4, it will play.
//By Spacefarer Games
//Contact jake@spacefarergames.com

#include "video_replacement.h"
#include "cdrom.h"
#include "gpu.h"
#include "host.h"
#include "system.h"

#include "util/audio_stream.h"
#include "util/gpu_device.h"
#include "util/imgui_manager.h"

#include "common/dynamic_library.h"
#include "common/error.h"
#include "common/file_system.h"
#include "common/log.h"
#include "common/path.h"
#include "common/timer.h"

#include "IconsFontAwesome.h"
#include "fmt/format.h"
#include "imgui.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <vector>

// FFmpeg dynamic loading - we load it at runtime instead of linking
// This is enabled by default - set to 0 to disable video replacement
#define WITH_FFMPEG 1

#if WITH_FFMPEG

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244) // conversion warnings
#endif

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/version.h>
#include <libavformat/avformat.h>
#include <libavformat/version.h>
#include <libavutil/opt.h>
#include <libavutil/version.h>
#include <libswresample/swresample.h>
#include <libswresample/version.h>
#include <libswscale/swscale.h>
#include <libswscale/version.h>
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

// Define the function imports we need
#define VISIT_AVCODEC_IMPORTS(X)                                                                                       \
  X(avcodec_find_decoder)                                                                                              \
  X(avcodec_alloc_context3)                                                                                            \
  X(avcodec_parameters_to_context)                                                                                     \
  X(avcodec_open2)                                                                                                     \
  X(avcodec_free_context)                                                                                              \
  X(avcodec_send_packet)                                                                                               \
  X(avcodec_receive_frame)                                                                                             \
  X(avcodec_flush_buffers)                                                                                             \
  X(av_packet_alloc)                                                                                                   \
  X(av_packet_free)                                                                                                    \
  X(av_packet_unref)

#define VISIT_AVFORMAT_IMPORTS(X)                                                                                      \
  X(avformat_open_input)                                                                                               \
  X(avformat_close_input)                                                                                              \
  X(avformat_find_stream_info)                                                                                         \
  X(av_find_best_stream)                                                                                               \
  X(av_read_frame)                                                                                                     \
  X(av_seek_frame)

#define VISIT_AVUTIL_IMPORTS(X)                                                                                        \
  X(av_frame_alloc)                                                                                                    \
  X(av_frame_free)                                                                                                     \
  X(av_frame_get_buffer)                                                                                               \
  X(av_strerror)                                                                                                       \
  X(av_opt_set_int)                                                                                                    \
  X(av_opt_set_sample_fmt)                                                                                             \
  X(av_samples_get_buffer_size)                                                                                        \
  X(av_get_bytes_per_sample)

// Note: av_q2d is a static inline function in FFmpeg headers, not an exported symbol
// We implement it ourselves below

#define VISIT_SWSCALE_IMPORTS(X)                                                                                       \
  X(sws_getContext)                                                                                                    \
  X(sws_scale)                                                                                                         \
  X(sws_freeContext)

#define VISIT_SWRESAMPLE_IMPORTS(X)                                                                                    \
  X(swr_alloc)                                                                                                         \
  X(swr_init)                                                                                                          \
  X(swr_free)                                                                                                          \
  X(swr_convert)

#endif // WITH_FFMPEG

LOG_CHANNEL(VideoReplacement);

namespace VideoReplacement {

namespace {

// Target playback framerate - videos play at this fixed rate regardless of source FPS
static constexpr double TARGET_FPS = 30.0;
static constexpr double FRAME_DURATION = 1.0 / TARGET_FPS;

// Returns true if this STR file's replacement video has its own audio track that should be played.
// SONY.STR and BEHAVIER.STR are new videos with audio tracks.
// For all other STR files (including INTRO.STR), the replacement video audio is muted
// and only the visual overlay is shown while the game's original audio continues.
static bool ShouldPlayReplacementAudio(std::string_view str_file_path)
{
  std::string filename = std::string(Path::GetFileName(str_file_path));
  INFO_LOG("ShouldPlayReplacementAudio: original filename = '{}'", filename);

  // Convert to uppercase for case-insensitive comparison
  for (char& c : filename)
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

  INFO_LOG("ShouldPlayReplacementAudio: uppercase filename = '{}'", filename);

  // SONY.STR and BEHAVIER.STR are new videos with audio - play their audio
  // All other STRs (including INTRO.STR) are visual-only overlays - mute replacement audio
  bool result = (filename == "SONY.STR" || filename == "BEHAVIER.STR");
  INFO_LOG("ShouldPlayReplacementAudio: result = {}", result);
  return result;
}

struct State
{
  std::string game_serial;
  std::string game_path;
  std::string replacements_directory;

  // Current replacement video state
  std::string current_str_path;
  std::string current_mp4_path;
  u32 str_file_size_sectors = 0;
  u32 str_start_lba = 0;
  u32 str_end_lba = 0;
  u32 sectors_outside_str = 0;  // Count of consecutive sectors read outside STR range

  // Track last played STR to avoid immediate re-trigger while still in same file
  std::string last_played_str_path;
  u32 last_played_end_lba = 0;

#if WITH_FFMPEG
  AVFormatContext* format_ctx = nullptr;
  AVCodecContext* video_codec_ctx = nullptr;
  AVCodecContext* audio_codec_ctx = nullptr;
  int video_stream_index = -1;
  int audio_stream_index = -1;
  AVFrame* frame = nullptr;
  AVFrame* rgb_frame = nullptr;
  AVFrame* audio_frame = nullptr;
  AVPacket* packet = nullptr;
  SwsContext* sws_ctx = nullptr;
  SwrContext* swr_ctx = nullptr;
  u8* video_buffer = nullptr;
  int video_buffer_size = 0;

  // Audio resampling output buffer
  u8* audio_buffer = nullptr;
  int audio_buffer_size = 0;
  int audio_sample_rate = 0;
  int audio_channels = 0;
#endif

  // Separate audio stream for replacement video audio (plays directly through system audio)
  std::unique_ptr<AudioStream> audio_stream;

  // Ring buffer for audio samples
  std::vector<s16> audio_ring_buffer;
  std::atomic<size_t> audio_write_pos{0};
  std::atomic<size_t> audio_read_pos{0};
  static constexpr size_t AUDIO_RING_BUFFER_SIZE = 44100 * 2 * 2;  // 2 seconds of stereo audio
  std::mutex audio_mutex;

  // Texture for rendering video
  std::unique_ptr<GPUTexture> video_texture;

  // Playback state
  bool is_playing = false;
  bool was_paused_before_video = false;
  bool should_play_replacement_audio = false; // True for SONY.STR and BEHAVIER.STR (play their audio)
  Timer playback_timer;
  double video_duration = 0.0;
  double video_fps = 0.0;
  double last_frame_time = 0.0;  // Real time when we last decoded a frame
  int start_frame = 0;           // Frame offset when starting mid-video (for seeking)
  int current_frame = 0;         // Current frame number
  int total_frames = 0;          // Total frames in video
  int video_width = 0;
  int video_height = 0;
};

static State s_state;

// Audio source for video replacement - reads from ring buffer and plays through system audio
class VideoAudioSource : public AudioStreamSource
{
public:
  void ReadFrames(SampleType* samples, u32 num_frames) override
  {
    const u32 num_samples = num_frames * 2;  // stereo

    std::lock_guard<std::mutex> lock(s_state.audio_mutex);

    size_t buffer_size = s_state.audio_ring_buffer.size();
    if (buffer_size == 0)
    {
      // Ring buffer not initialized, output silence
      std::memset(samples, 0, num_samples * sizeof(SampleType));
      return;
    }

    size_t read_pos = s_state.audio_read_pos.load();
    size_t write_pos = s_state.audio_write_pos.load();
    u32 samples_read = 0;

    for (u32 i = 0; i < num_samples; i++)
    {
      if (read_pos != write_pos)
      {
        samples[i] = s_state.audio_ring_buffer[read_pos];
        read_pos = (read_pos + 1) % buffer_size;
        samples_read++;
      }
      else
      {
        // Buffer underrun, output silence for remaining samples
        samples[i] = 0;
      }
    }

    s_state.audio_read_pos.store(read_pos);

    if (samples_read > 0)
    {
      VERBOSE_LOG("AudioSource: read {} samples from ring buffer", samples_read);
    }
  }
};

static VideoAudioSource s_audio_source;

#if WITH_FFMPEG

// Function pointer wrappers for dynamically loaded FFmpeg functions
#define DECLARE_IMPORT(X) static inline decltype(X)* wrap_##X;
VISIT_AVCODEC_IMPORTS(DECLARE_IMPORT);
VISIT_AVFORMAT_IMPORTS(DECLARE_IMPORT);
VISIT_AVUTIL_IMPORTS(DECLARE_IMPORT);
VISIT_SWSCALE_IMPORTS(DECLARE_IMPORT);
VISIT_SWRESAMPLE_IMPORTS(DECLARE_IMPORT);
#undef DECLARE_IMPORT

// Dynamic library handles
static DynamicLibrary s_avcodec_library;
static DynamicLibrary s_avformat_library;
static DynamicLibrary s_avutil_library;
static DynamicLibrary s_swscale_library;
static DynamicLibrary s_swresample_library;
static bool s_library_loaded = false;
static std::mutex s_load_mutex;
static std::mutex s_playback_mutex;  // Protects playback state from concurrent access

// Helper function: av_q2d is a static inline in FFmpeg headers, so we implement it ourselves
static inline double rational_to_double(AVRational a)
{
  return a.num / (double)a.den;
}

static bool LoadFFmpeg(Error* error);
static void UnloadFFmpeg();
static bool OpenVideoFile(const char* path, Error* error);
static void CloseVideoFile();
static bool DecodeNextFrame(Error* error);
static bool SeekToTimestamp(double timestamp_seconds, Error* error);
static bool UpdateVideoTexture(Error* error);

#endif // WITH_FFMPEG

} // namespace

} // namespace VideoReplacement

namespace VideoReplacement {

void Initialize()
{
  INFO_LOG("Video replacement system initialized");
}

void Shutdown()
{
#if WITH_FFMPEG
  CloseVideoFile();
  UnloadFFmpeg();
#endif

  // Stop audio stream if running
  if (s_state.audio_stream)
  {
    Error stop_error;
    s_state.audio_stream->Stop(&stop_error);
    s_state.audio_stream.reset();
  }

  // Clear ring buffer
  {
    std::lock_guard<std::mutex> lock(s_state.audio_mutex);
    s_state.audio_ring_buffer.clear();
    s_state.audio_write_pos.store(0);
    s_state.audio_read_pos.store(0);
  }

  // Reset state (can't use assignment due to atomics)
  s_state.game_serial.clear();
  s_state.game_path.clear();
  s_state.replacements_directory.clear();
  s_state.current_str_path.clear();
  s_state.current_mp4_path.clear();
  s_state.last_played_str_path.clear();
  s_state.video_texture.reset();
  s_state.is_playing = false;
  s_state.should_play_replacement_audio = false;

  INFO_LOG("Video replacement system shut down");
}

void GameChanged(std::string_view game_serial, std::string_view game_path)
{
#if WITH_FFMPEG
  CloseVideoFile();
#endif

  s_state.game_serial = std::string(game_serial);
  s_state.game_path = std::string(game_path);

  // Build the replacements directory path
  // Look for videos in: <user_directory>/videos/<game_serial>/
  if (!game_serial.empty())
  {
    s_state.replacements_directory =
      Path::Combine(EmuFolders::UserResources, fmt::format("videos/{}", game_serial));

    if (FileSystem::DirectoryExists(s_state.replacements_directory.c_str()))
    {
      INFO_LOG("Video replacement directory found: {}", s_state.replacements_directory);
      
      // List all MP4 files in the directory for debugging
      FileSystem::FindResultsArray mp4_files;
      FileSystem::FindFiles(s_state.replacements_directory.c_str(), "*.mp4", FILESYSTEM_FIND_FILES, &mp4_files);
      
      if (!mp4_files.empty())
      {
        INFO_LOG("Found {} replacement video(s):", mp4_files.size());
        for (const auto& file : mp4_files)
        {
          INFO_LOG("  - {}", file.FileName);
        }
      }
      else
      {
        WARNING_LOG("Video replacement directory exists but contains no MP4 files");
      }
    }
    else
    {
      INFO_LOG("No video replacement directory found at: {}", s_state.replacements_directory);
      INFO_LOG("Create this directory and add MP4 files to enable video replacement");
    }
  }
}

std::string GetReplacementPath(std::string_view str_file_path)
{
  if (s_state.replacements_directory.empty())
  {
    VERBOSE_LOG("No replacements directory configured");
    return {};
  }

  if (str_file_path.empty())
  {
    VERBOSE_LOG("Empty STR file path provided");
    return {};
  }

  // Convert the STR path to a potential MP4 path
  // Example: "MOVIE/INTRO.STR" -> "INTRO.mp4"
  std::string filename = std::string(Path::GetFileName(str_file_path));

  // Remove extension and add .mp4
  size_t dot_pos = filename.find_last_of('.');
  if (dot_pos != std::string::npos)
    filename = filename.substr(0, dot_pos);

  filename += ".mp4";

  std::string mp4_path = Path::Combine(s_state.replacements_directory, filename);

  INFO_LOG("Looking for replacement video: {} -> {}", str_file_path, mp4_path);

  if (FileSystem::FileExists(mp4_path.c_str()))
  {
    INFO_LOG("Found replacement video: {}", mp4_path);
    return mp4_path;
  }

  INFO_LOG("No replacement video found at: {}", mp4_path);
  return {};
}

bool IsPlayingReplacement()
{
  return s_state.is_playing;
}

bool ShouldPlayReplacementAudio()
{
  return s_state.is_playing && s_state.should_play_replacement_audio;
}

bool OnSTRFileOpened(std::string_view str_file_path, u32 file_size_sectors, u32 start_lba, u32 end_lba, u32 current_lba)
{
  INFO_LOG("STR file opened: {} ({} sectors, LBA {}-{}, current LBA {})", str_file_path, file_size_sectors, start_lba, end_lba, current_lba);

#if !WITH_FFMPEG
  // FFmpeg not available, feature is not implemented yet
  WARNING_LOG("Video replacement is not enabled (FFmpeg not compiled in)");
  return false;
#else
  // Don't re-trigger for the same STR file we just finished playing
  // This prevents re-starting the video when we're still reading the tail end of the STR
  // But allow it to play again if we've moved past its LBA range and come back
  if (s_state.last_played_str_path == str_file_path && 
      start_lba == s_state.str_start_lba && end_lba == s_state.str_end_lba)
  {
    VERBOSE_LOG("Skipping re-trigger for recently played STR: {}", str_file_path);
    return false;
  }

  // Check if replacement exists
  std::string mp4_path = GetReplacementPath(str_file_path);
  if (mp4_path.empty())
  {
    INFO_LOG("No replacement video available for: {}", str_file_path);
    return false;
  }

  INFO_LOG("Starting video replacement overlay: {} ({} sectors) -> {}", str_file_path, file_size_sectors, mp4_path);

  // Load FFmpeg if not already loaded
  INFO_LOG("Loading FFmpeg libraries for video playback...");
  Error error;
  if (!LoadFFmpeg(&error))
  {
    ERROR_LOG("Failed to load FFmpeg libraries: {}", error.GetDescription());
    Host::AddIconOSDMessage(OSDMessageType::Quick, "VideoReplacementFFmpegError", ICON_FA_EXCLAMATION,
                            fmt::format("Failed to load FFmpeg libraries for video replacement.\n{}",
                                        error.GetDescription()));
    return false;
  }
  INFO_LOG("FFmpeg libraries loaded successfully");

  // Store state including LBA range for tracking
  s_state.current_str_path = std::string(str_file_path);
  s_state.current_mp4_path = mp4_path;
  s_state.str_file_size_sectors = file_size_sectors;
  s_state.str_start_lba = start_lba;
  s_state.str_end_lba = end_lba;
  s_state.sectors_outside_str = 0;


  // Open and start playing the MP4
  INFO_LOG("Opening replacement video file: {}", mp4_path);
  if (!OpenVideoFile(mp4_path.c_str(), &error))
  {
    ERROR_LOG("Failed to open replacement video {}: {}", mp4_path, error.GetDescription());
    return false;
  }

  s_state.total_frames = static_cast<int>(s_state.video_duration * s_state.video_fps);

  // Calculate starting position based on how far into the STR file we are
  // This syncs the HD video to where the game is actually reading
  double start_time = 0.0;
  int start_frame = 0;
  if (current_lba > start_lba && file_size_sectors > 0)
  {
    const u32 sectors_into_file = current_lba - start_lba;
    const double progress = static_cast<double>(sectors_into_file) / static_cast<double>(file_size_sectors);
    start_time = progress * s_state.video_duration;
    start_frame = static_cast<int>(progress * s_state.total_frames);
    start_frame = std::min(start_frame, s_state.total_frames - 1);
    INFO_LOG("STR read position: {} sectors in ({:.1f}%), starting video at {:.2f}s (frame {}/{})", 
             sectors_into_file, progress * 100.0, start_time, start_frame, s_state.total_frames);
  }

  s_state.is_playing = true;
  s_state.playback_timer.Reset();
  s_state.last_frame_time = 0.0;
  s_state.start_frame = start_frame;  // Remember starting position for target_frame calculation
  s_state.current_frame = start_frame;

  // Check if this is a video that should play its own audio (SONY.STR, BEHAVIER.STR)
  // Game audio is NEVER muted - it always plays
  // For SONY.STR and BEHAVIER.STR, we also play the replacement video's audio
  // For all other STRs (including INTRO.STR), replacement video audio is muted (visual overlay only)
  s_state.should_play_replacement_audio = ShouldPlayReplacementAudio(str_file_path);

  INFO_LOG("Audio state: should_play={}, audio_stream_idx={}, audio_codec={}, swr={}, audio_frame={}, audio_buffer={}",
           s_state.should_play_replacement_audio, s_state.audio_stream_index,
           s_state.audio_codec_ctx ? "yes" : "no", s_state.swr_ctx ? "yes" : "no",
           s_state.audio_frame ? "yes" : "no", s_state.audio_buffer ? "yes" : "no");

  if (s_state.should_play_replacement_audio)
  {
    INFO_LOG("Video has audio track - will play replacement audio (SONY.STR or BEHAVIER.STR)");

    // Initialize ring buffer and create audio stream for playback
    {
      std::lock_guard<std::mutex> lock(s_state.audio_mutex);
      s_state.audio_ring_buffer.resize(State::AUDIO_RING_BUFFER_SIZE);
      std::fill(s_state.audio_ring_buffer.begin(), s_state.audio_ring_buffer.end(), 0);
      s_state.audio_write_pos.store(0);
      s_state.audio_read_pos.store(0);
    }

    // Create a separate audio stream for video replacement audio
    // This plays directly through the system audio, bypassing the emulator's SPU
    Error audio_error;
    s_state.audio_stream = AudioStream::CreateStream(
      AudioStream::DEFAULT_BACKEND,
      44100,  // sample rate
      2,      // channels (stereo)
      2048,   // output latency frames
      false,  // minimal latency
      {},     // default driver
      {},     // default device
      &s_audio_source,
      true,   // auto start
      &audio_error);

    if (s_state.audio_stream)
    {
      INFO_LOG("Created separate audio stream for video replacement audio");
    }
    else
    {
      WARNING_LOG("Failed to create audio stream for video replacement: {}", audio_error.GetDescription());
    }
  }
  else
  {
    INFO_LOG("Video is visual overlay only - replacement audio muted, game audio continues");
  }

  // Seek to the starting position if needed
  if (start_time > 0.1)  // Only seek if more than 100ms in
  {
    INFO_LOG("Seeking video to {:.2f}s (frame {})", start_time, start_frame);
    Error seek_error;
    if (!SeekToTimestamp(start_time, &seek_error))
    {
      WARNING_LOG("Failed to seek to {:.2f}s: {} - starting from beginning", start_time, seek_error.GetDescription());
      s_state.start_frame = 0;
      s_state.current_frame = 0;
    }
    else
    {
      // Decode one frame after seeking to update the texture
      if (!DecodeNextFrame(&seek_error))
      {
        WARNING_LOG("Failed to decode frame after seek: {}", seek_error.GetDescription());
      }
      INFO_LOG("Seeked to approximately frame {}", s_state.current_frame);
    }
  }
  else
  {
    // Starting from beginning, decode first frame
    s_state.start_frame = 0;
    Error decode_error;
    if (!DecodeNextFrame(&decode_error))
    {
      WARNING_LOG("Failed to decode first frame: {}", decode_error.GetDescription());
    }
  }

  // Game keeps running normally - we just overlay the HD video on top
  // Game audio always plays. For SONY.STR and BEHAVIER.STR, replacement video audio also plays.
  // For all other STRs, replacement video audio is muted (visual overlay only).
  INFO_LOG("Video overlay started - game audio continues, replacement audio: {}", 
           s_state.should_play_replacement_audio ? "playing" : "muted");
  INFO_LOG("Video details: {}x{} @ {:.2f} fps, duration: {:.2f}s, {} total frames (playing at {:.0f} FPS), starting at frame {}", 
           s_state.video_width, s_state.video_height,
           s_state.video_fps, s_state.video_duration, s_state.total_frames, TARGET_FPS, s_state.current_frame);

  Host::AddIconOSDMessage(OSDMessageType::Quick, "VideoReplacement", ICON_FA_FILM,
                          fmt::format("HD video overlay: {}", Path::GetFileName(str_file_path)));

  return true;
#endif
}

void OnSTRFileClosed()
{
#if WITH_FFMPEG
  std::unique_lock<std::mutex> lock(s_playback_mutex);

  if (!s_state.is_playing)
    return;

  INFO_LOG("STR file closed - stopping video overlay (was playing: {})", s_state.current_str_path);

  // Remember what we just played to avoid immediate re-trigger
  s_state.last_played_str_path = s_state.current_str_path;
  s_state.last_played_end_lba = s_state.str_end_lba;

  // Stop and destroy the audio stream if it was created
  if (s_state.audio_stream)
  {
    Error stop_error;
    s_state.audio_stream->Stop(&stop_error);
    s_state.audio_stream.reset();
    INFO_LOG("Stopped video replacement audio stream");
  }

  // Clear the ring buffer
  {
    std::lock_guard<std::mutex> audio_lock(s_state.audio_mutex);
    s_state.audio_ring_buffer.clear();
    s_state.audio_write_pos.store(0);
    s_state.audio_read_pos.store(0);
  }

  s_state.is_playing = false;
  s_state.current_str_path.clear();
  s_state.current_mp4_path.clear();
  s_state.str_start_lba = 0;
  s_state.str_end_lba = 0;
  s_state.sectors_outside_str = 0;
  s_state.should_play_replacement_audio = false;

  lock.unlock();  // Release lock before calling CloseVideoFile
  CloseVideoFile();
#endif
}

void OnSectorRead(u32 lba)
{
#if WITH_FFMPEG
  // Clear "last played" memory if we've moved well past that STR file
  // This allows the same video to play again if the game loops back to it
  if (!s_state.last_played_str_path.empty() && lba > s_state.last_played_end_lba + 100)
  {
    VERBOSE_LOG("Cleared last played STR memory (moved past LBA {})", s_state.last_played_end_lba);
    s_state.last_played_str_path.clear();
    s_state.last_played_end_lba = 0;
  }

  if (!s_state.is_playing)
    return;
  
  // Check if this sector is within the STR file's LBA range
  if (lba >= s_state.str_start_lba && lba <= s_state.str_end_lba)
  {
    // Still reading STR file, reset counter
    s_state.sectors_outside_str = 0;
  }
  else
  {
    // Reading outside STR range
    s_state.sectors_outside_str++;
    
    // If we've read several sectors outside the STR range, the game has moved on
    // Use a small threshold to avoid false positives from seek operations
    static constexpr u32 OUTSIDE_THRESHOLD = 5;
    if (s_state.sectors_outside_str >= OUTSIDE_THRESHOLD)
    {
      INFO_LOG("Detected {} consecutive sectors outside STR range (LBA {} not in {}-{}), stopping overlay",
               s_state.sectors_outside_str, lba, s_state.str_start_lba, s_state.str_end_lba);
      OnSTRFileClosed();
    }
  }
#endif
}

bool ProcessReplacementFrame()
{
#if !WITH_FFMPEG
  return false;
#else
  std::unique_lock<std::mutex> lock(s_playback_mutex);

  if (!s_state.is_playing)
    return false;

  // Safety check: ensure video context is still valid
  if (!s_state.format_ctx || !s_state.video_codec_ctx)
  {
    WARNING_LOG("Video context lost during playback, stopping");
    s_state.is_playing = false;
    lock.unlock();  // Release lock before calling CloseVideoFile which may also lock
    CloseVideoFile();
    return false;
  }

  // Get elapsed time and check if we've played all frames
  const double elapsed_time = s_state.playback_timer.GetTimeSeconds();

  // Calculate which frame we should be showing at 30 FPS
  // Add start_frame offset to account for seeking into the middle of a video
  const int target_frame = s_state.start_frame + static_cast<int>(elapsed_time / FRAME_DURATION);

  // Check if video is finished (played all frames or target exceeds total)
  if (s_state.total_frames > 0 && (s_state.current_frame >= s_state.total_frames || target_frame >= s_state.total_frames))
  {
    INFO_LOG("Video overlay finished - played {} frames in {:.2f}s", 
             s_state.current_frame, elapsed_time);

    s_state.is_playing = false;
    lock.unlock();  // Release lock before calling CloseVideoFile
    CloseVideoFile();

    INFO_LOG("Video overlay complete");
    return false;
  }

  // Decode frames to catch up with target (skip frames if we're behind to stay in sync with audio)
  // Limit how many frames we try to decode in one iteration to prevent hanging under load
  static constexpr int MAX_FRAMES_PER_UPDATE = 5;
  int frames_decoded = 0;

  while (s_state.current_frame < target_frame && s_state.current_frame < s_state.total_frames)
  {
    // Safety check: ensure contexts are still valid (could be cleared by another thread)
    if (!s_state.format_ctx || !s_state.video_codec_ctx)
    {
      WARNING_LOG("Video context lost during frame decode, stopping playback");
      s_state.is_playing = false;
      return false;
    }

    Error error;
    if (!DecodeNextFrame(&error))
    {
      // End of stream or error - stop overlay
      VERBOSE_LOG("Video decode finished or error: {}", error.GetDescription());
      s_state.is_playing = false;
      lock.unlock();  // Release lock before calling CloseVideoFile
      CloseVideoFile();
      return false;
    }
    s_state.current_frame++;
    frames_decoded++;

    // Limit frames decoded per update to avoid blocking for too long under heavy load
    if (frames_decoded >= MAX_FRAMES_PER_UPDATE)
    {
      const int frames_behind = target_frame - s_state.current_frame;
      if (frames_behind > 0)
      {
        VERBOSE_LOG("Decoded {} frames, still {} behind - will catch up next update", frames_decoded, frames_behind);
      }
      break;
    }

    // Log if we're skipping multiple frames (falling behind)
    const int frames_behind = target_frame - s_state.current_frame;
    if (frames_behind > 2)
    {
      VERBOSE_LOG("Video behind by {} frames, skipping to catch up", frames_behind);
    }
  }

  s_state.last_frame_time = elapsed_time;
  return true;
#endif
}

void RenderReplacementFrame()
{
#if WITH_FFMPEG
  if (!s_state.is_playing || !s_state.video_texture)
    return;

  // Render fullscreen video - simplified approach
  // Note: This draws the video as a fullscreen quad
  // TODO: Use proper ImGui rendering or background callback when available
  
  (void)s_state; // Prevent unused warning
  
  // For now, we'll just keep the texture updated and let the system handle rendering
  // Future: Implement proper fullscreen video overlay rendering
#endif
}

GPUTexture* GetCurrentTexture()
{
  std::lock_guard<std::mutex> lock(s_playback_mutex);
  return s_state.video_texture.get();
}

} // namespace VideoReplacement

#if WITH_FFMPEG

namespace VideoReplacement {
namespace {

static bool LoadFFmpeg(Error* error)
{
  std::unique_lock lock(s_load_mutex);
  if (s_library_loaded)
    return true;

  // Try to load FFmpeg libraries with multiple version attempts
  static constexpr auto open_dynlib = [](DynamicLibrary& lib, const char* name, int major_version, Error* error) {
    // Try exact version first
    std::string full_name = DynamicLibrary::GetVersionedFilename(name, major_version);
    if (lib.Open(full_name.c_str(), error))
    {
      INFO_LOG("Loaded {}", full_name);
      return true;
    }
    
    // Try compatible versions (one major version older/newer)
    INFO_LOG("Exact version {} not found, trying compatible versions...", full_name);
    for (int version_offset = -1; version_offset <= 1; version_offset++)
    {
      if (version_offset == 0) continue; // Already tried exact
      
      int try_version = major_version + version_offset;
      if (try_version < 0) continue;
      
      full_name = DynamicLibrary::GetVersionedFilename(name, try_version);
      Error temp_error;
      if (lib.Open(full_name.c_str(), &temp_error))
      {
        WARNING_LOG("Loaded {} (expected version {}, compatibility not guaranteed)", full_name, major_version);
        return true;
      }
    }
    
    // Try without version suffix as last resort
    full_name = std::string(name);
#ifdef _WIN32
    full_name += ".dll";
#elif defined(__APPLE__)
    full_name = "lib" + full_name + ".dylib";
#else
    full_name = "lib" + full_name + ".so";
#endif
    
    if (lib.Open(full_name.c_str(), error))
    {
      WARNING_LOG("Loaded {} (version unknown, compatibility not guaranteed)", full_name);
      return true;
    }
    
    ERROR_LOG("Failed to load {} with any version", name);
    return false;
  };

  // Load libraries
  if (!open_dynlib(s_avutil_library, "avutil", LIBAVUTIL_VERSION_MAJOR, error))
  {
    UnloadFFmpeg();
    return false;
  }
  
  if (!open_dynlib(s_avcodec_library, "avcodec", LIBAVCODEC_VERSION_MAJOR, error))
  {
    UnloadFFmpeg();
    return false;
  }
  
  if (!open_dynlib(s_avformat_library, "avformat", LIBAVFORMAT_VERSION_MAJOR, error))
  {
    UnloadFFmpeg();
    return false;
  }
  
  if (!open_dynlib(s_swscale_library, "swscale", LIBSWSCALE_VERSION_MAJOR, error))
  {
    UnloadFFmpeg();
    return false;
  }
  
  if (!open_dynlib(s_swresample_library, "swresample", LIBSWRESAMPLE_VERSION_MAJOR, error))
  {
    UnloadFFmpeg();
    return false;
  }

  INFO_LOG("All FFmpeg DLLs loaded, resolving symbols...");

  // Resolve symbols - ALL must succeed or we fail
  bool all_symbols_found = true;
  
#define RESOLVE_IMPORT(X) \
  if (!s_avcodec_library.GetSymbol(#X, &wrap_##X)) { \
    ERROR_LOG("Failed to resolve critical avcodec symbol: {}", #X); \
    all_symbols_found = false; \
  }
  VISIT_AVCODEC_IMPORTS(RESOLVE_IMPORT);
#undef RESOLVE_IMPORT

#define RESOLVE_IMPORT(X) \
  if (!s_avformat_library.GetSymbol(#X, &wrap_##X)) { \
    ERROR_LOG("Failed to resolve critical avformat symbol: {}", #X); \
    all_symbols_found = false; \
  }
  VISIT_AVFORMAT_IMPORTS(RESOLVE_IMPORT);
#undef RESOLVE_IMPORT

#define RESOLVE_IMPORT(X) \
  if (!s_avutil_library.GetSymbol(#X, &wrap_##X)) { \
    ERROR_LOG("Failed to resolve critical avutil symbol: {}", #X); \
    all_symbols_found = false; \
  }
  VISIT_AVUTIL_IMPORTS(RESOLVE_IMPORT);
#undef RESOLVE_IMPORT

#define RESOLVE_IMPORT(X) \
  if (!s_swscale_library.GetSymbol(#X, &wrap_##X)) { \
    ERROR_LOG("Failed to resolve critical swscale symbol: {}", #X); \
    all_symbols_found = false; \
  }
  VISIT_SWSCALE_IMPORTS(RESOLVE_IMPORT);
#undef RESOLVE_IMPORT

#define RESOLVE_IMPORT(X) \
  if (!s_swresample_library.GetSymbol(#X, &wrap_##X)) { \
    ERROR_LOG("Failed to resolve critical swresample symbol: {}", #X); \
    all_symbols_found = false; \
  }
  VISIT_SWRESAMPLE_IMPORTS(RESOLVE_IMPORT);
#undef RESOLVE_IMPORT

  if (!all_symbols_found)
  {
    ERROR_LOG("Failed to resolve all required FFmpeg symbols");
    UnloadFFmpeg();
    Error::SetString(error, "FFmpeg DLLs loaded but symbols could not be resolved. The version may be incompatible.");
    return false;
  }

  s_library_loaded = true;
  std::atexit(&UnloadFFmpeg);
  INFO_LOG("FFmpeg libraries loaded successfully for video replacement");
  return true;
}

static void UnloadFFmpeg()
{
#define CLEAR_IMPORT(X) wrap_##X = nullptr;
  VISIT_AVCODEC_IMPORTS(CLEAR_IMPORT);
  VISIT_AVFORMAT_IMPORTS(CLEAR_IMPORT);
  VISIT_AVUTIL_IMPORTS(CLEAR_IMPORT);
  VISIT_SWSCALE_IMPORTS(CLEAR_IMPORT);
  VISIT_SWRESAMPLE_IMPORTS(CLEAR_IMPORT);
#undef CLEAR_IMPORT

  s_swresample_library.Close();
  s_swscale_library.Close();
  s_avutil_library.Close();
  s_avformat_library.Close();
  s_avcodec_library.Close();
  s_library_loaded = false;

  INFO_LOG("FFmpeg libraries unloaded");
}

static bool OpenVideoFile(const char* path, Error* error)
{
  CloseVideoFile();

  // Open video file
  INFO_LOG("Opening video file with FFmpeg: {}", path);
  if (wrap_avformat_open_input(&s_state.format_ctx, path, nullptr, nullptr) != 0)
  {
    Error::SetStringFmt(error, "Failed to open video file: {}", path);
    ERROR_LOG("FFmpeg avformat_open_input failed for: {}", path);
    return false;
  }

  // Retrieve stream information
  if (wrap_avformat_find_stream_info(s_state.format_ctx, nullptr) < 0)
  {
    Error::SetString(error, "Failed to find stream information");
    CloseVideoFile();
    return false;
  }

  // Find video stream
  INFO_LOG("Finding video stream in file");
  s_state.video_stream_index = wrap_av_find_best_stream(s_state.format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (s_state.video_stream_index < 0)
  {
    Error::SetString(error, "No video stream found");
    ERROR_LOG("No video stream found in file: {}", path);
    CloseVideoFile();
    return false;
  }
  INFO_LOG("Found video stream at index: {}", s_state.video_stream_index);

  AVStream* video_stream = s_state.format_ctx->streams[s_state.video_stream_index];

  // Find decoder
  INFO_LOG("Looking for video codec decoder (codec_id: {})", static_cast<int>(video_stream->codecpar->codec_id));
  const AVCodec* codec = wrap_avcodec_find_decoder(video_stream->codecpar->codec_id);
  if (!codec)
  {
    Error::SetString(error, "Video codec not supported");
    ERROR_LOG("Video codec not supported (codec_id: {})", static_cast<int>(video_stream->codecpar->codec_id));
    CloseVideoFile();
    return false;
  }
  INFO_LOG("Found decoder: {}", codec->name);

  // Create codec context
  s_state.video_codec_ctx = wrap_avcodec_alloc_context3(codec);
  if (!s_state.video_codec_ctx)
  {
    Error::SetString(error, "Failed to allocate codec context");
    CloseVideoFile();
    return false;
  }

  if (wrap_avcodec_parameters_to_context(s_state.video_codec_ctx, video_stream->codecpar) < 0)
  {
    Error::SetString(error, "Failed to copy codec parameters");
    CloseVideoFile();
    return false;
  }

  // Open codec
  if (wrap_avcodec_open2(s_state.video_codec_ctx, codec, nullptr) < 0)
  {
    Error::SetString(error, "Failed to open codec");
    CloseVideoFile();
    return false;
  }

  // Store video properties
  s_state.video_width = s_state.video_codec_ctx->width;
  s_state.video_height = s_state.video_codec_ctx->height;
  s_state.video_fps = rational_to_double(video_stream->avg_frame_rate);
  s_state.video_duration = static_cast<double>(s_state.format_ctx->duration) / AV_TIME_BASE;

  INFO_LOG("Successfully opened video file:");
  INFO_LOG("  Resolution: {}x{}", s_state.video_width, s_state.video_height);
  INFO_LOG("  Frame rate: {:.2f} fps", s_state.video_fps);
  INFO_LOG("  Duration: {:.2f} seconds", s_state.video_duration);
  INFO_LOG("  Codec: {}", codec->name);

  // Allocate frames and packet
  s_state.frame = wrap_av_frame_alloc();
  s_state.rgb_frame = wrap_av_frame_alloc();
  s_state.packet = wrap_av_packet_alloc();

  if (!s_state.frame || !s_state.rgb_frame || !s_state.packet)
  {
    Error::SetString(error, "Failed to allocate frame/packet");
    CloseVideoFile();
    return false;
  }

  // Create RGB frame buffer
  s_state.rgb_frame->format = AV_PIX_FMT_RGBA;
  s_state.rgb_frame->width = s_state.video_width;
  s_state.rgb_frame->height = s_state.video_height;

  if (wrap_av_frame_get_buffer(s_state.rgb_frame, 0) < 0)
  {
    Error::SetString(error, "Failed to allocate RGB frame buffer");
    CloseVideoFile();
    return false;
  }

  // Create swscale context for color conversion
  // Validate dimensions are reasonable before creating context
  if (s_state.video_width <= 0 || s_state.video_height <= 0 || 
      s_state.video_width > 8192 || s_state.video_height > 8192)
  {
    Error::SetStringFmt(error, "Invalid video dimensions: {}x{}", s_state.video_width, s_state.video_height);
    CloseVideoFile();
    return false;
  }

  s_state.sws_ctx =
    wrap_sws_getContext(s_state.video_width, s_state.video_height, s_state.video_codec_ctx->pix_fmt,
                        s_state.video_width, s_state.video_height, AV_PIX_FMT_RGBA, SWS_BILINEAR, nullptr, nullptr, nullptr);

  if (!s_state.sws_ctx)
  {
    Error::SetString(error, "Failed to create swscale context");
    CloseVideoFile();
    return false;
  }

  // Create GPU texture for rendering
  INFO_LOG("Creating GPU texture for video rendering ({}x{})", s_state.video_width, s_state.video_height);

  // Validate GPU device is available
  if (!g_gpu_device)
  {
    Error::SetString(error, "GPU device not available");
    ERROR_LOG("Cannot create video texture: GPU device is null");
    CloseVideoFile();
    return false;
  }

  Error tex_error;
  s_state.video_texture = g_gpu_device->FetchTexture(s_state.video_width, s_state.video_height, 1, 1, 1,
                                                      GPUTexture::Type::Texture, GPUTextureFormat::RGBA8,
                                                      GPUTexture::Flags::None, nullptr, 0, &tex_error);

  if (!s_state.video_texture)
  {
    Error::SetStringFmt(error, "Failed to create video texture: {}", tex_error.GetDescription());
    ERROR_LOG("Failed to create GPU texture: {}", tex_error.GetDescription());
    CloseVideoFile();
    return false;
  }
  INFO_LOG("GPU texture created successfully");

  // Set up audio stream if this video should play audio (SONY.STR, BEHAVIER.STR)
  // We check this later when we know the STR path, but set up the decoder now if audio exists
  s_state.audio_stream_index = wrap_av_find_best_stream(s_state.format_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
  if (s_state.audio_stream_index >= 0)
  {
    INFO_LOG("Found audio stream at index: {}", s_state.audio_stream_index);

    AVStream* audio_stream = s_state.format_ctx->streams[s_state.audio_stream_index];
    const AVCodec* audio_codec = wrap_avcodec_find_decoder(audio_stream->codecpar->codec_id);

    if (audio_codec)
    {
      INFO_LOG("Found audio decoder: {}", audio_codec->name);

      s_state.audio_codec_ctx = wrap_avcodec_alloc_context3(audio_codec);
      if (s_state.audio_codec_ctx)
      {
        if (wrap_avcodec_parameters_to_context(s_state.audio_codec_ctx, audio_stream->codecpar) >= 0)
        {
          if (wrap_avcodec_open2(s_state.audio_codec_ctx, audio_codec, nullptr) >= 0)
          {
            // Get audio properties
            s_state.audio_sample_rate = s_state.audio_codec_ctx->sample_rate;
            s_state.audio_channels = s_state.audio_codec_ctx->ch_layout.nb_channels;

            INFO_LOG("Audio stream: {} Hz, {} channels, codec: {}, sample_fmt: {}", 
                     s_state.audio_sample_rate, s_state.audio_channels, audio_codec->name,
                     static_cast<int>(s_state.audio_codec_ctx->sample_fmt));

            // Allocate audio frame
            s_state.audio_frame = wrap_av_frame_alloc();
            if (!s_state.audio_frame)
            {
              WARNING_LOG("Failed to allocate audio frame");
            }
            else
            {
              // Set up resampler to convert to 44100 Hz stereo s16 (SPU format)
              s_state.swr_ctx = wrap_swr_alloc();
              if (s_state.swr_ctx)
              {
                // Determine input channel layout
                int64_t in_ch_layout = AV_CH_LAYOUT_STEREO;
                if (s_state.audio_codec_ctx->ch_layout.order == AV_CHANNEL_ORDER_NATIVE && 
                    s_state.audio_codec_ctx->ch_layout.u.mask != 0)
                {
                  in_ch_layout = s_state.audio_codec_ctx->ch_layout.u.mask;
                }
                else if (s_state.audio_channels == 1)
                {
                  in_ch_layout = AV_CH_LAYOUT_MONO;
                }

                INFO_LOG("Setting up resampler: in_ch_layout=0x{:X}, in_rate={}, in_fmt={}", 
                         in_ch_layout, s_state.audio_sample_rate, static_cast<int>(s_state.audio_codec_ctx->sample_fmt));

                // Set input parameters using 'ich'/'icl' style names (older FFmpeg compatibility)
                int ret = 0;
                ret |= wrap_av_opt_set_int(s_state.swr_ctx, "in_channel_layout", in_ch_layout, 0);
                ret |= wrap_av_opt_set_int(s_state.swr_ctx, "in_sample_rate", s_state.audio_sample_rate, 0);
                ret |= wrap_av_opt_set_sample_fmt(s_state.swr_ctx, "in_sample_fmt", s_state.audio_codec_ctx->sample_fmt, 0);

                // Set output parameters (44100 Hz stereo s16 for SPU)
                ret |= wrap_av_opt_set_int(s_state.swr_ctx, "out_channel_layout", AV_CH_LAYOUT_STEREO, 0);
                ret |= wrap_av_opt_set_int(s_state.swr_ctx, "out_sample_rate", 44100, 0);
                ret |= wrap_av_opt_set_sample_fmt(s_state.swr_ctx, "out_sample_fmt", AV_SAMPLE_FMT_S16, 0);

                if (ret < 0)
                {
                  WARNING_LOG("Failed to set resampler options (ret={})", ret);
                }

                int init_ret = wrap_swr_init(s_state.swr_ctx);
                if (init_ret >= 0)
                {
                  // Allocate audio buffer (enough for 1 second of audio)
                  s_state.audio_buffer_size = 44100 * 2 * sizeof(s16);  // 1 sec at 44100 Hz stereo s16
                  s_state.audio_buffer = static_cast<u8*>(std::malloc(s_state.audio_buffer_size));

                  INFO_LOG("Audio resampler initialized successfully: {} Hz {} ch -> 44100 Hz stereo s16", 
                           s_state.audio_sample_rate, s_state.audio_channels);
                }
                else
                {
                  char err_buf[256];
                  wrap_av_strerror(init_ret, err_buf, sizeof(err_buf));
                  WARNING_LOG("Failed to initialize audio resampler: {} ({})", err_buf, init_ret);
                  wrap_swr_free(&s_state.swr_ctx);
                }
              }
              else
              {
                WARNING_LOG("Failed to allocate SwrContext");
              }
            }
          }
          else
          {
            WARNING_LOG("Failed to open audio codec");
            wrap_avcodec_free_context(&s_state.audio_codec_ctx);
          }
        }
        else
        {
          WARNING_LOG("Failed to copy audio codec parameters");
          wrap_avcodec_free_context(&s_state.audio_codec_ctx);
        }
      }
    }
    else
    {
      WARNING_LOG("Audio codec not supported (codec_id: {})", static_cast<int>(audio_stream->codecpar->codec_id));
    }
  }
  else
  {
    INFO_LOG("No audio stream found in video file");
  }

  return true;
}

static void CloseVideoFile()
{
  INFO_LOG("Closing video file and cleaning up resources");

  // Release GPU texture first while GPU device is still valid
  if (s_state.video_texture)
  {
    if (g_gpu_device)
      s_state.video_texture.reset();
    else
      s_state.video_texture.release(); // Don't try to free if device is gone
  }

  // Check if FFmpeg is still loaded before calling cleanup functions
  if (!s_library_loaded)
  {
    WARNING_LOG("FFmpeg already unloaded, skipping cleanup calls");
    // Just clear the pointers
    s_state.sws_ctx = nullptr;
    s_state.swr_ctx = nullptr;
    s_state.rgb_frame = nullptr;
    s_state.audio_frame = nullptr;
    s_state.frame = nullptr;
    s_state.packet = nullptr;
    s_state.video_codec_ctx = nullptr;
    s_state.audio_codec_ctx = nullptr;
    s_state.format_ctx = nullptr;
    s_state.video_stream_index = -1;
    s_state.audio_stream_index = -1;
    s_state.video_width = 0;
    s_state.video_height = 0;
    s_state.video_fps = 0.0;
    s_state.video_duration = 0.0;
    if (s_state.audio_buffer)
    {
      std::free(s_state.audio_buffer);
      s_state.audio_buffer = nullptr;
    }
    s_state.audio_buffer_size = 0;
    s_state.audio_sample_rate = 0;
    s_state.audio_channels = 0;
    return;
  }

  if (s_state.sws_ctx)
  {
    wrap_sws_freeContext(s_state.sws_ctx);
    s_state.sws_ctx = nullptr;
  }

  if (s_state.swr_ctx)
  {
    wrap_swr_free(&s_state.swr_ctx);
  }

  if (s_state.rgb_frame)
  {
    wrap_av_frame_free(&s_state.rgb_frame);
  }

  if (s_state.audio_frame)
  {
    wrap_av_frame_free(&s_state.audio_frame);
  }

  if (s_state.frame)
  {
    wrap_av_frame_free(&s_state.frame);
  }

  if (s_state.packet)
  {
    wrap_av_packet_free(&s_state.packet);
  }

  if (s_state.video_codec_ctx)
  {
    wrap_avcodec_free_context(&s_state.video_codec_ctx);
  }

  if (s_state.audio_codec_ctx)
  {
    wrap_avcodec_free_context(&s_state.audio_codec_ctx);
  }

  if (s_state.format_ctx)
  {
    wrap_avformat_close_input(&s_state.format_ctx);
  }

  if (s_state.audio_buffer)
  {
    std::free(s_state.audio_buffer);
    s_state.audio_buffer = nullptr;
  }

  s_state.video_stream_index = -1;
  s_state.audio_stream_index = -1;
  s_state.video_width = 0;
  s_state.video_height = 0;
  s_state.video_fps = 0.0;
  s_state.video_duration = 0.0;
  s_state.audio_buffer_size = 0;
  s_state.audio_sample_rate = 0;
  s_state.audio_channels = 0;
}

static bool SeekToTimestamp(double timestamp_seconds, Error* error)
{
  if (!s_state.format_ctx || s_state.video_stream_index < 0)
  {
    Error::SetString(error, "No video file open");
    return false;
  }

  if (!s_state.video_codec_ctx)
  {
    Error::SetString(error, "Video codec not initialized");
    return false;
  }

  // Validate stream index is within bounds
  if (s_state.video_stream_index >= static_cast<int>(s_state.format_ctx->nb_streams))
  {
    Error::SetString(error, "Invalid video stream index");
    return false;
  }

  // Convert timestamp to stream timebase
  AVStream* video_stream = s_state.format_ctx->streams[s_state.video_stream_index];
  if (!video_stream)
  {
    Error::SetString(error, "Video stream not available");
    return false;
  }

  int64_t target_ts = static_cast<int64_t>(timestamp_seconds * AV_TIME_BASE);
  
  // Seek to the nearest keyframe before the target timestamp
  // AVSEEK_FLAG_BACKWARD seeks to keyframe at or before the timestamp
  int ret = wrap_av_seek_frame(s_state.format_ctx, -1, target_ts, AVSEEK_FLAG_BACKWARD);
  if (ret < 0)
  {
    char err_buf[256];
    wrap_av_strerror(ret, err_buf, sizeof(err_buf));
    Error::SetStringFmt(error, "Failed to seek to {:.2f}s: {}", timestamp_seconds, err_buf);
    return false;
  }

  // Flush the codec buffers after seeking
  wrap_avcodec_flush_buffers(s_state.video_codec_ctx);
  if (s_state.audio_codec_ctx)
    wrap_avcodec_flush_buffers(s_state.audio_codec_ctx);

  INFO_LOG("Seeked video to {:.2f}s", timestamp_seconds);
  return true;
}

static bool DecodeNextFrame(Error* error)
{
  // Validate required objects are available
  if (!s_state.format_ctx || !s_state.video_codec_ctx || !s_state.packet || 
      !s_state.frame || !s_state.rgb_frame || !s_state.sws_ctx)
  {
    Error::SetString(error, "Video decoder not initialized");
    return false;
  }

  // Validate stream index
  if (s_state.video_stream_index < 0 || 
      s_state.video_stream_index >= static_cast<int>(s_state.format_ctx->nb_streams))
  {
    Error::SetString(error, "Invalid video stream index");
    return false;
  }

  // Read packets until we get a video frame
  int read_result;
  while ((read_result = wrap_av_read_frame(s_state.format_ctx, s_state.packet)) >= 0)
  {
    // Handle audio packets if we should play replacement audio
    if (s_state.packet->stream_index == s_state.audio_stream_index && 
        s_state.should_play_replacement_audio && s_state.audio_codec_ctx && 
        s_state.swr_ctx && s_state.audio_frame && s_state.audio_buffer)
    {
      // Send audio packet to decoder
      int ret = wrap_avcodec_send_packet(s_state.audio_codec_ctx, s_state.packet);
      if (ret >= 0)
      {
        // Receive decoded audio frames
        int recv_ret;
        while ((recv_ret = wrap_avcodec_receive_frame(s_state.audio_codec_ctx, s_state.audio_frame)) >= 0)
        {
          VERBOSE_LOG("Decoded audio frame: {} samples", s_state.audio_frame->nb_samples);
          // Calculate max output samples (buffer size / bytes per sample / channels)
          int max_out_samples = s_state.audio_buffer_size / (2 * sizeof(s16));  // stereo s16

          // Resample audio to 44100 Hz stereo s16
          u8* out_buffer = s_state.audio_buffer;
          int out_samples = wrap_swr_convert(s_state.swr_ctx,
                                              &out_buffer, max_out_samples,
                                              (const u8**)s_state.audio_frame->extended_data, 
                                              s_state.audio_frame->nb_samples);

          VERBOSE_LOG("swr_convert: in={} samples, out={} samples", s_state.audio_frame->nb_samples, out_samples);

          if (out_samples > 0)
          {
            // Write audio samples to ring buffer for playback through separate audio stream
            std::lock_guard<std::mutex> lock(s_state.audio_mutex);

            if (!s_state.audio_ring_buffer.empty())
            {
              const s16* src = reinterpret_cast<const s16*>(s_state.audio_buffer);
              const u32 num_samples = out_samples * 2;  // stereo
              size_t write_pos = s_state.audio_write_pos.load();
              size_t buffer_size = s_state.audio_ring_buffer.size();

              for (u32 i = 0; i < num_samples; i++)
              {
                s_state.audio_ring_buffer[write_pos] = src[i];
                write_pos = (write_pos + 1) % buffer_size;
              }

              s_state.audio_write_pos.store(write_pos);
              VERBOSE_LOG("Wrote {} audio samples to ring buffer", num_samples);
            }
          }
        }
      }
      wrap_av_packet_unref(s_state.packet);
      continue;  // Keep reading until we get a video frame
    }

    if (s_state.packet->stream_index == s_state.video_stream_index)
    {
      // Send packet to decoder
      int ret = wrap_avcodec_send_packet(s_state.video_codec_ctx, s_state.packet);
      wrap_av_packet_unref(s_state.packet);

      if (ret < 0)
      {
        char err_buf[256];
        wrap_av_strerror(ret, err_buf, sizeof(err_buf));
        Error::SetStringFmt(error, "Failed to send packet to decoder: {}", err_buf);
        return false;
      }

      // Receive decoded frame
      ret = wrap_avcodec_receive_frame(s_state.video_codec_ctx, s_state.frame);
      if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
      {
        continue;
      }
      else if (ret < 0)
      {
        char err_buf[256];
        wrap_av_strerror(ret, err_buf, sizeof(err_buf));
        Error::SetStringFmt(error, "Failed to receive frame from decoder: {}", err_buf);
        return false;
      }

      // Validate frame data before processing
      if (!s_state.frame->data[0])
      {
        Error::SetString(error, "Decoded frame has no data");
        return false;
      }

      // Validate rgb_frame is still valid (could have been cleared)
      if (!s_state.rgb_frame || !s_state.rgb_frame->data[0])
      {
        Error::SetString(error, "RGB frame buffer not available");
        return false;
      }

      // Convert to RGB
      int scale_ret = wrap_sws_scale(s_state.sws_ctx, s_state.frame->data, s_state.frame->linesize, 0, s_state.video_height,
                     s_state.rgb_frame->data, s_state.rgb_frame->linesize);

      if (scale_ret <= 0)
      {
        Error::SetString(error, "Failed to scale frame");
        return false;
      }

      // Update texture
      return UpdateVideoTexture(error);
    }

    wrap_av_packet_unref(s_state.packet);
  }

  // End of file or error
  if (read_result == AVERROR_EOF)
  {
    Error::SetString(error, "End of video file");
  }
  else
  {
    char err_buf[256];
    wrap_av_strerror(read_result, err_buf, sizeof(err_buf));
    Error::SetStringFmt(error, "Error reading frame: {}", err_buf);
  }
  return false;
}

static bool UpdateVideoTexture(Error* error)
{
  if (!s_state.video_texture)
  {
    Error::SetString(error, "Video texture not available");
    return false;
  }

  if (!s_state.rgb_frame || !s_state.rgb_frame->data[0])
  {
    Error::SetString(error, "No RGB frame data available");
    return false;
  }

  if (s_state.video_width <= 0 || s_state.video_height <= 0)
  {
    Error::SetString(error, "Invalid video dimensions");
    return false;
  }

  // Upload RGB data to GPU texture
  s_state.video_texture->Update(0, 0, s_state.video_width, s_state.video_height, s_state.rgb_frame->data[0],
                                s_state.rgb_frame->linesize[0]);

  return true;
}

} // namespace
} // namespace VideoReplacement

#endif // WITH_FFMPEG
