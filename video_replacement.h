//Video Replacement Header

#pragma once

#include "common/types.h"

#include <memory>
#include <string>
#include <string_view>

class Error;
class GPUTexture;

namespace VideoReplacement {

/// Initializes the video replacement system.
void Initialize();

/// Shuts down the video replacement system.
void Shutdown();

/// Called when a new game is loaded.
void GameChanged(std::string_view game_serial, std::string_view game_path);

/// Checks if a replacement video exists for the given STR file path.
/// Returns the path to the MP4 file if found, otherwise an empty string.
std::string GetReplacementPath(std::string_view str_file_path);

/// Checks if we're currently playing a replacement video.
bool IsPlayingReplacement();

/// Checks if the current replacement video should play its own audio.
/// Returns true for videos like SONY.STR and BEHAVIER.STR which have new audio tracks.
/// Returns false for silent video overlays where game audio should be muted.
bool ShouldPlayReplacementAudio();

/// Called when the CD-ROM detects an STR file is being read.
/// This will start the HD video overlay while the game plays underneath.
/// current_lba is where we're currently reading, used to calculate video start time.
/// Returns true if a replacement video was found and started.
bool OnSTRFileOpened(std::string_view str_file_path, u32 file_size_sectors, u32 start_lba, u32 end_lba, u32 current_lba);

/// Called when the STR file stops being read (skipped or ended).
/// This stops the HD video overlay.
void OnSTRFileClosed();

/// Called on each sector read to detect when STR playback has stopped.
/// If reading moves outside the STR file range, stops the overlay.
void OnSectorRead(u32 lba);

/// Called each frame while playing a replacement video.
/// Returns true if playback should continue, false if finished.
bool ProcessReplacementFrame();

/// Renders the current video frame to the screen.
void RenderReplacementFrame();

/// Gets the current video texture (for rendering).
GPUTexture* GetCurrentTexture();

} // namespace VideoReplacement
