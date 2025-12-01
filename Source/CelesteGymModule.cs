using System;
using Celeste;
using Celeste.Mod;
using Microsoft.Xna.Framework;
using Monocle;

namespace Celeste.Mod.CelesteGym;

/// <summary>
/// Main module for CelesteGym - RL training environment for Celeste.
/// Implements fast-forward via multiple Update() calls per render frame,
/// NOT via Engine.TimeRate modification (which breaks physics).
/// </summary>
public class CelesteGymModule : EverestModule {

    public static CelesteGymModule Instance { get; private set; } = null!;

    public override Type SettingsType => typeof(CelesteGymModuleSettings);
    public static CelesteGymModuleSettings Settings => (CelesteGymModuleSettings) Instance._Settings;

    public override Type SessionType => typeof(CelesteGymModuleSession);
    public static CelesteGymModuleSession Session => (CelesteGymModuleSession) Instance._Session;

    public override Type SaveDataType => typeof(CelesteGymModuleSaveData);
    public static CelesteGymModuleSaveData SaveData => (CelesteGymModuleSaveData) Instance._SaveData;

    // Shared memory bridge
    private SharedMemoryBridge? sharedMemory;
    
    // Game state tracking
    private GameState currentState;
    private uint updateCount = 0;
    private uint renderFrameCount = 0;
    
    // Fast-forward configuration
    private bool fastForwardEnabled = true;
    private int updatesPerFrame = 1;  // 400x speedup target
    
    public CelesteGymModule() {
        Instance = this;
#if DEBUG
        Logger.SetLogLevel(nameof(CelesteGymModule), LogLevel.Verbose);
#else
        Logger.SetLogLevel(nameof(CelesteGymModule), LogLevel.Info);
#endif
    }

    public override void Load() {
        // Initialize shared memory
        sharedMemory = new SharedMemoryBridge();
        if (!sharedMemory.Initialize()) {
            Logger.Log(LogLevel.Error, "CelesteGym", "Failed to initialize shared memory!");
            return;
        }
        
        fastForwardEnabled = Settings.FastForwardEnabled;
        updatesPerFrame = Settings.UpdatesPerFrame; 
        // Hook Celeste.Update to run multiple times per render frame
        // This is how we achieve speedup without breaking physics
        On.Celeste.Celeste.Update += FastForwardUpdate;

        //Replace hardware input with virtual input:.
        On.Monocle.MInput.Update += OverrideMInputUpdate;

        // Hook level update to extract state
        Everest.Events.Level.OnBeforeUpdate += OnLevelUpdate;
            
        // Hook level load for initialization
        Everest.Events.Level.OnLoadLevel += OnLoadLevel;


        Logger.Log(LogLevel.Info, "CelesteGym", 
            $"Module loaded - fast-forward: {(fastForwardEnabled ? updatesPerFrame + "x" : "disabled")}");
    }

    public override void Unload() {
        On.Celeste.Celeste.Update -= FastForwardUpdate;
        On.Monocle.MInput.Update -= OverrideMInputUpdate;
        Everest.Events.Level.OnBeforeUpdate -= OnLevelUpdate;
        Everest.Events.Level.OnLoadLevel -= OnLoadLevel;

        sharedMemory?.Dispose();
        sharedMemory = null;
        
        Logger.Log(LogLevel.Info, "CelesteGym", "Module unloaded");
    }

    private static void OverrideMInputUpdate(On.Monocle.MInput.orig_Update orig){
        Level? level = Engine.Scene as Level;
        if (level != null) {
            // update cached action from Python
            // InputController.action = 4;//Instance.sharedMemory.ReadAction();

            InputController.action = Instance.sharedMemory.ReadAction();
            // Now propagate to virtual buttons
            orig();
            InputController.ApplyAction();
            MInput.UpdateVirtualInputs();  // ← Need reflection to call this
            
            Logger.Log(LogLevel.Info, "CelesteGym",  $"After update - Jump pressed: {Input.Jump.Pressed}");
        }else{
            orig();
        }
    }
    /// <summary>
    /// Main fast-forward hook. Runs Update() multiple times per render frame.
    /// This achieves speedup while maintaining proper physics integration.
    /// </summary>
    private static void FastForwardUpdate(
        On.Celeste.Celeste.orig_Update orig,
        global::Celeste.Celeste self,
        GameTime gameTime
    ) {
        Instance.renderFrameCount++;
        
        // Determine how many updates to run this frame
        Level? level = Engine.Scene as Level;
        int updates = (Instance.fastForwardEnabled && level != null && !level.Paused) ? Instance.updatesPerFrame : 1;
        
        // Run multiple game updates per render frame
        for (int i = 0; i < updates; i++) {
            try {
                // Call original Update with normal delta time
                // TimeRate remains 1.0, so physics runs correctly
                orig(self, gameTime);
                
                // OnLevelUpdate will be called during orig() if in a level
                
            } catch (Exception ex) {
                Logger.Log(LogLevel.Error, "CelesteGym", 
                    $"Exception during update: {ex.Message}\n{ex.StackTrace}");
                
                // Disable fast-forward on error to allow debugging
                Instance.fastForwardEnabled = false;
                break;
            }
        }
        if (Settings.StateLoggingInterval > 0 && 
        Instance.currentState.FrameCount % Settings.StateLoggingInterval == 0) {
            Logger.Log(LogLevel.Verbose, "CelesteGym", 
                $"Frame {Instance.currentState.FrameCount}: " +
                $"Pos=({Instance.currentState.PosX:F1}, {Instance.currentState.PosY:F1}) " +
                $"Action={InputController.action}");
        }
        // Periodic performance logging
        if (Instance.renderFrameCount % 600 == 0) {  // Every 10 seconds at 60fps
            float actualSpeedup = Instance.updateCount / (float)Instance.renderFrameCount;
            Logger.Log(LogLevel.Info, "CelesteGym", 
                $"Performance: {actualSpeedup:F1}x speedup " +
                $"({Instance.updateCount} updates / {Instance.renderFrameCount} frames)");
        }
    }

    /// <summary>
    /// Called when a level is loaded. Use for initialization.
    /// </summary>
    private static void OnLoadLevel(Level level, Player.IntroTypes playerIntro, bool isFromLoader) {
        // Reset frame counters for this episode
        Instance.currentState.FrameCount = 0;
        
        Logger.Log(LogLevel.Info, "CelesteGym", 
            $"Level loaded: {level.Session.Level}");
    }
    
    /// <summary>
    /// Called before each level update. Extract state, communicate with Python.
    /// This gets called updatesPerFrame times per render frame.
    /// </summary>
    private static void OnLevelUpdate(Level level) {
        if (Instance.sharedMemory == null) return;
        
        Instance.updateCount++;
        
        Player player = level.Tracker.GetEntity<Player>();
        if (player == null) return;
        
        // Extract current game state
        ExtractState(level, player);
        
        // Write state to shared memory (Python will read when ready)
        Instance.sharedMemory.WriteState(ref Instance.currentState);
        
        // Periodic debug logging
        if (Instance.currentState.FrameCount % 2400 == 0) {  // Every ~10 seconds at 400x
            Logger.Log(LogLevel.Verbose, "CelesteGym", 
                $"Frame {Instance.currentState.FrameCount}: " +
                $"Pos=({Instance.currentState.PosX:F1}, {Instance.currentState.PosY:F1}) " +
                $"Action={InputController.action}");
        }
    }

    /// <summary>
    /// Extract all relevant game state into the GameState struct.
    /// </summary>
    private static void ExtractState(Level level, Player player) {
        // Player physics state
        Instance.currentState.PosX = player.Position.X;
        Instance.currentState.PosY = player.Position.Y;
        Instance.currentState.VelX = player.Speed.X;
        Instance.currentState.VelY = player.Speed.Y;
        Instance.currentState.Stamina = player.Stamina;
        
        // Player abilities
        Instance.currentState.Dashes = (byte)player.Dashes;
        Instance.currentState.OnGround = (byte)(player.OnGround() ? 1 : 0);
        Instance.currentState.Dead = (byte)(player.Dead ? 1 : 0);
        /*Instance.currentState.WallSlideDir = (byte)(
            player.WallSlideDir == -1 ? 255 :  // -1 as unsigned byte
            player.WallSlideDir == 1 ? 1 : 0
        );*/
        Instance.currentState.Facing = (byte)(player.Facing == Facings.Left ? 255 : 1);
        
        // Frame counter
        Instance.currentState.FrameCount++;
        
        // Build local observation grid
        unsafe {
            fixed (byte* gridPtr = Instance.currentState.LocalGrid) {
                GridManager.BuildLocalGrid(level, player, gridPtr);
                
                // Periodic grid debugging
                if (Instance.currentState.FrameCount % 6000 == 0) {
                    string gridDump = GridManager.DumpGrid(gridPtr);
                    Logger.Log(LogLevel.Verbose, "CelesteGym", 
                        $"Grid at frame {Instance.currentState.FrameCount}:\n{gridDump}");
                }
            }
        }
    }

    /// <summary>
    /// Enable or disable fast-forward mode.
    /// </summary>
    public void SetFastForward(bool enabled, int? speedup = null) {
        fastForwardEnabled = enabled;
        if (speedup.HasValue) {
            updatesPerFrame = speedup.Value;
        }
        
        Logger.Log(LogLevel.Info, "CelesteGym", 
            $"Fast-forward: {(enabled ? updatesPerFrame + "x" : "disabled")}");
    }

    /// <summary>
    /// Get current performance metrics.
    /// </summary>
    public (uint updates, uint frames, float speedup) GetMetrics() {
        float speedup = renderFrameCount > 0 ? updateCount / (float)renderFrameCount : 0;
        return (updateCount, renderFrameCount, speedup);
    }
}