"""
Shared memory bridge for communication with Celeste game.
Reads game state and writes actions using double-buffered memory.

Memory Layout (2120 bytes):
    [0-1055]     Buffer A (GameState)
    [1056-2111]  Buffer B (GameState)
    [2112-2115]  Write Index (uint32)
    [2116-2117]  Action (ushort)
    [2118-2119]  Reserved
"""

import struct
import mmap
import time
from typing import Optional, Tuple
import numpy as np

class GameState:
    """
    Represents the game state read from shared memory.
    Must match the C# GameState struct layout exactly.
    """
    
    # Struct format: matches C# layout exactly
    # f = float (4 bytes)
    # B = unsigned byte (1 byte)
    # H = unsigned short (2 bytes)
    # I = unsigned int (4 bytes)
    # 1024s = 1024 bytes (grid)
    STRUCT_FORMAT = '=fffffBBBBBBBBI1024s'
    STRUCT_SIZE = 1056
    
    def __init__(self, data: bytes):
        """Parse game state from raw bytes."""
        unpacked = struct.unpack(self.STRUCT_FORMAT, data)
        
        # Player state (32 bytes)
        self.pos_x = unpacked[0]
        self.pos_y = unpacked[1]
        self.vel_x = unpacked[2]
        self.vel_y = unpacked[3]
        self.stamina = unpacked[4]

        self.dashes = unpacked[5]
        self.on_ground = bool(unpacked[6])
        self.wall_slide_dir = unpacked[7]
        self.facing = unpacked[8]
        self.state = unpacked[9]
        self.dead = bool(unpacked[10])
        self.can_dash = bool(unpacked[11])
        # padding: unpacked[12]
        self.frame_count = unpacked[13]
        
        # Local grid (1024 bytes)
        self.local_grid = np.frombuffer(unpacked[14], dtype=np.uint8).reshape(32, 32)
    
    def __repr__(self):
        return (f"GameState(pos=({self.pos_x:.1f}, {self.pos_y:.1f}), "
                f"vel=({self.vel_x:.2f}, {self.vel_y:.2f}), "
                f"dashes={self.dashes}, on_ground={self.on_ground}, "
                f"frame={self.frame_count})")


class SharedMemoryBridge:
    """
    Thread-safe shared memory bridge for reading game state and writing actions.
    Uses double-buffering to avoid races and ensure consistent observations.
    """
    
    BUFFER_SIZE = 1056
    BUFFER_A_OFFSET = 0
    BUFFER_B_OFFSET = 1056
    WRITE_INDEX_OFFSET = 2112
    ACTION_OFFSET = 2116
    TOTAL_SIZE = 2120
    
    SHARED_MEMORY_NAME = "CelesteGymSharedMemory"
    
    def __init__(self):
        self.shm: Optional[mmap.mmap] = None
        self.last_write_index = 0
        self._opened = False
    
    def open(self, timeout_sec: float = 5.0) -> bool:
        """
        Open shared memory. Waits for C# side to create it.
        
        Args:
            timeout_sec: How long to wait for shared memory to be created
            
        Returns:
            True if opened successfully, False otherwise
        """
        import platform
        
        start_time = time.time()
        
        while time.time() - start_time < timeout_sec:
            try:
                if platform.system() == "Windows":
                    # Windows: Use mmap with named shared memory
                    import ctypes
                    from ctypes import wintypes
                    
                    # Open existing file mapping
                    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
                    
                    GENERIC_READ = 0x80000000
                    GENERIC_WRITE = 0x40000000
                    FILE_MAP_ALL_ACCESS = 0xF001F
                    
                    # handle = kernel32.OpenFileMappingW(
                    #     FILE_MAP_ALL_ACCESS,
                    #     False,
                    #     self.SHARED_MEMORY_NAME
                    # )
                    
                    # if not handle:
                    #     raise OSError(f"Could not open shared memory: {ctypes.get_last_error()}")
                    
                    # # Map view of file
                    # addr = kernel32.MapViewOfFile(
                    #     handle,
                    #     FILE_MAP_ALL_ACCESS,
                    #     0,
                    #     0,
                    #     self.TOTAL_SIZE
                    # )
                    
                    # if not addr:
                    #     kernel32.CloseHandle(handle)
                    #     raise OSError(f"Could not map view: {ctypes.get_last_error()}")
                    
                    # Create mmap object from address
                    self.shm = mmap.mmap(-1, self.TOTAL_SIZE, access=mmap.ACCESS_WRITE, tagname=self.SHARED_MEMORY_NAME)
                    
                    # Note: This is a simplified version. Full implementation would need
                    # to properly wrap the Windows handle. For now, we'll use a workaround.
                    
                else:
                    # Linux/Mac: Use POSIX shared memory
                    import posix_ipc
                    
                    shm_obj = posix_ipc.SharedMemory(
                        f"/{self.SHARED_MEMORY_NAME}",
                        flags=0,  # Open existing
                        size=self.TOTAL_SIZE
                    )
                    
                    self.shm = mmap.mmap(
                        shm_obj.fd,
                        self.TOTAL_SIZE,
                        mmap.MAP_SHARED,
                        mmap.PROT_READ | mmap.PROT_WRITE
                    )
                    
                    shm_obj.close_fd()
                
                self._opened = True
                print(f"Shared memory opened: {self.TOTAL_SIZE} bytes")
                return True
                
            except (FileNotFoundError, OSError) as e:
                # Not created yet, wait and retry
                time.sleep(0.1)
                continue
        
        print(f"Timeout waiting for shared memory after {timeout_sec}s")
        return False
    
    def read_state(self) -> Optional[GameState]:
        """
        Read the latest game state from shared memory.
        Uses double-buffering to ensure consistent reads.
        
        Returns:
            GameState object or None if error
        """
        if not self._opened or self.shm is None:
            return None
        
        try:
            # Read current write index
            self.shm.seek(self.WRITE_INDEX_OFFSET)
            write_idx = struct.unpack('=I', self.shm.read(4))[0]
            
            # Choose stable buffer (opposite of what C# is writing to)
            # If write_idx is odd, C# writes to A (offset 0), so read from B (offset 1056)
            # If write_idx is even, C# writes to B (offset 1056), so read from A (offset 0)
            buffer_offset = self.BUFFER_B_OFFSET if (write_idx % 2 == 1) else self.BUFFER_A_OFFSET
            
            # Read from stable buffer
            self.shm.seek(buffer_offset)
            state_bytes = self.shm.read(self.BUFFER_SIZE)
            
            self.last_write_index = write_idx
            
            return GameState(state_bytes)
            
        except Exception as e:
            print(f"Error reading state: {e}")
            return None
    
    def write_action(self, action: int) -> bool:
        """
        Write action to shared memory.
        
        Args:
            action: Action ID (0-14)
            
        Returns:
            True if written successfully
        """
        if not self._opened or self.shm is None:
            return False
        
        try:
            # Validate action
            if not (0 <= action <= 14):
                print(f"Warning: Invalid action {action}, clamping to [0, 14]")
                action = max(0, min(14, action))
            
            # Write action (ushort = 2 bytes)
            self.shm.seek(self.ACTION_OFFSET)
            self.shm.write(struct.pack('=H', action))
            
            return True
            
        except Exception as e:
            print(f"Error writing action: {e}")
            return False
    
    def get_write_index(self) -> int:
        """Get current write index (for debugging)."""
        if not self._opened or self.shm is None:
            return -1
        
        try:
            self.shm.seek(self.WRITE_INDEX_OFFSET)
            return struct.unpack('=I', self.shm.read(4))[0]
        except:
            return -1
    
    def close(self):
        """Close shared memory."""
        if self.shm is not None:
            self.shm.close()
            self.shm = None
            self._opened = False
            print("Shared memory closed")
    
    def __enter__(self):
        """Context manager entry."""
        if not self._opened:
            self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Test utilities
def test_connection(duration_sec: float = 5.0):
    """
    Test shared memory connection by reading states continuously.
    
    Args:
        duration_sec: How long to run the test
    """
    print("Testing shared memory connection...")
    
    bridge = SharedMemoryBridge()
    
    if not bridge.open(timeout_sec=10.0):
        print("Failed to open shared memory!")
        return
    
    print(f"Reading states for {duration_sec} seconds...")
    
    start_time = time.time()
    frame_count = 0
    last_frame = -1
    
    try:
        while time.time() - start_time < duration_sec:
            bridge.write_action(4)
            state = bridge.read_state()
            
            if state is not None:
                frame_count += 1
                
                # Check for frame updates
                if state.frame_count != last_frame:
                    last_frame = state.frame_count
                    
                    # Print every 60 frames (~1 sec at normal speed)
                    if frame_count % 60 == 0:
                        print(f"Frame {state.frame_count}: {state}")
                        
                        # Show grid preview (center 8x8)
                        grid_center = state.local_grid[12:20, 12:20]
                        print("Grid center:")
                        for row in grid_center:
                            print(''.join(['#' if cell == 1 else '.' for cell in row]))
            
            time.sleep(0.01)  # 100 Hz polling
        bridge.write_action(0)
    
    except KeyboardInterrupt:
        print("\nTest interrupted")
    
    finally:
        bridge.close()
    
    elapsed = time.time() - start_time
    print(f"\nTest complete: {frame_count} reads in {elapsed:.2f}s "
          f"({frame_count/elapsed:.1f} reads/sec)")


if __name__ == "__main__":
    # Run test when executed directly
    test_connection(duration_sec=5.0)
