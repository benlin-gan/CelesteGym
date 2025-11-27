using System;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Threading;
using Celeste.Mod;

namespace Celeste.Mod.CelesteGym;

/// <summary>
/// Thread-safe shared memory bridge for communication between Celeste and Python.
/// Uses double-buffering to avoid blocking and ensure consistent reads.
/// 
/// Memory Layout (2120 bytes):
///   [0-1055]     Buffer A (GameState)
///   [1056-2111]  Buffer B (GameState)  
///   [2112-2115]  Write Index (uint32)
///   [2116-2117]  Action (ushort)
///   [2118-2119]  Reserved
/// </summary>
public class SharedMemoryBridge : IDisposable {
    
    // Memory layout constants
    private const int BUFFER_SIZE = 1056;
    private const int BUFFER_A_OFFSET = 0;
    private const int BUFFER_B_OFFSET = 1056;
    private const int WRITE_INDEX_OFFSET = 2112;
    private const int ACTION_OFFSET = 2116;
    private const int TOTAL_SIZE = 2120;
    
    private const string SHARED_MEMORY_NAME = "CelesteGymSharedMemory";
    
    private MemoryMappedFile? mmf;
    private MemoryMappedViewAccessor? accessor;
    private volatile uint writeIndex = 0;
    private ushort currentAction = 0;
    
    private readonly object disposeLock = new object();
    private bool disposed = false;
    
    /// <summary>
    /// Initialize shared memory. Creates new or opens existing.
    /// </summary>

    public bool Initialize() {
        try {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) {
                // Windows: Use named shared memory
                mmf = MemoryMappedFile.CreateOrOpen(
                    SHARED_MEMORY_NAME, 
                    TOTAL_SIZE,
                    MemoryMappedFileAccess.ReadWrite
                );
                Logger.Log(LogLevel.Info, "CelesteGym", "Initialized shared memory (Windows)");
            } else {
                // Unix: Use file-backed memory
                string sharedMemoryPath = Path.Combine("/dev/shm", SHARED_MEMORY_NAME);
                
                // Always create/truncate the file
                using (var fs = new FileStream(sharedMemoryPath, FileMode.Create, FileAccess.ReadWrite, FileShare.ReadWrite)) {
                    fs.SetLength(TOTAL_SIZE);
                }
                
                mmf = MemoryMappedFile.CreateFromFile(
                    sharedMemoryPath,
                    FileMode.Open,
                    null,
                    TOTAL_SIZE,
                    MemoryMappedFileAccess.ReadWrite
                );
                
                Logger.Log(LogLevel.Info, "CelesteGym", $"Initialized shared memory: {sharedMemoryPath}");
            }
            
            accessor = mmf.CreateViewAccessor(0, TOTAL_SIZE);
            return true;
            
        } catch (Exception ex) {
            Logger.Log(LogLevel.Error, "CelesteGym", $"Failed to initialize: {ex.Message}");
            return false;
        }
    }
    /// <summary>
    /// Write game state to shared memory using double-buffering.
    /// Thread-safe and non-blocking.
    /// </summary>
    public void WriteState(ref GameState state) {
        if (accessor == null || disposed) {
            return;
        }
        
        try {
            writeIndex++; //modify local copy only, python still sees unincremented index in shared memory
            int bufferOffset = (writeIndex % 2 == 0) ? BUFFER_B_OFFSET : BUFFER_A_OFFSET;
            
            // Write state to the reserved buffer
            accessor.Write(bufferOffset, ref state);
            
            // Memory barrier ensures buffer write completes before index is visible
            Thread.MemoryBarrier();
            
            // Update the shared index
            accessor.Write(WRITE_INDEX_OFFSET, writeIndex);
            
        } catch (Exception ex) {
            Logger.Log(LogLevel.Error, "CelesteGym", 
                $"Error writing state: {ex.Message}");
        }
    }
    
    /// <summary>
    /// Read the latest action from Python.
    /// Returns 0 (no input) if no new action available.
    /// </summary>
    public ushort ReadAction() {
        if (accessor == null || disposed) {
            return 0;
        }
        
        try {
            // Read action (Python writes this atomically)
            ushort action = accessor.ReadUInt16(ACTION_OFFSET);
            currentAction = action;
            return action;
            
        } catch (Exception ex) {
            Logger.Log(LogLevel.Error, "CelesteGym", 
                $"Error reading action: {ex.Message}");
            return currentAction; // Return last known action
        }
    }
    
    /// <summary>
    /// Get current write index (for debugging).
    /// </summary>
    public uint GetWriteIndex() {
        return writeIndex;
    }
    
    /// <summary>
    /// Get current action (for debugging).
    /// </summary>
    public ushort GetCurrentAction() {
        return currentAction;
    }
    
    public void Dispose() {
        lock (disposeLock) {
            if (disposed) {
                return;
            }
            
            disposed = true;
            
            accessor?.Dispose();
            accessor = null;
            
            mmf?.Dispose();
            mmf = null;
            
            Logger.Log(LogLevel.Info, "CelesteGym", "Shared memory disposed");
        }
    }
}

/// <summary>
/// Extension methods for MemoryMappedViewAccessor to read/write structs.
/// </summary>
public static class AccessorExtensions {
    public static void Write<T>(this MemoryMappedViewAccessor accessor, long position, ref T structure) 
        where T : struct {
        accessor.Write(position, ref structure);
    }
    
    public static void Read<T>(this MemoryMappedViewAccessor accessor, long position, out T structure) 
        where T : struct {
        accessor.Read(position, out structure);
    }
}