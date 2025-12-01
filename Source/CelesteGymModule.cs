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
    private int skippedUpdates = 0;
    
    
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
            $"Module loaded - fast-forward: {(Settings.FastForwardEnabled ? Settings.UpdatesPerFrame + "x" : "disabled")}");
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
        if (level != null && !level.Paused) {

            // InputController.action = Instance.sharedMemory.ReadAction();
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
        
        Level? level = Engine.Scene as Level;
        
        if (level == null || level.Paused || !Settings.FastForwardEnabled) {
            orig(self, gameTime);
            return;
        }
        
        // Even spacing across 16.67ms frame
        const double FRAME_TIME_MS = 16.67;
        double timePerUpdateSlot = FRAME_TIME_MS / Settings.UpdatesPerFrame;  // ~41.67μs
        
        var frameStart = System.Diagnostics.Stopwatch.GetTimestamp();
        double ticksPerMs = System.Diagnostics.Stopwatch.Frequency / 1000.0;
        
        int updatesExecuted = 0;
        
        for (int i = 0; i < Settings.UpdatesPerFrame; i++) {
            // Wait for this update slot's scheduled time
            double targetTimeMs = i * timePerUpdateSlot;
            
            while (true) {
                double elapsedMs = (System.Diagnostics.Stopwatch.GetTimestamp() - frameStart) / ticksPerMs;
                if (elapsedMs >= targetTimeMs) break;
                System.Threading.Thread.SpinWait(10);
            }
            
            // Check if Python has provided a NEW action
            if (Instance.sharedMemory.HasNewAction()) {
                // Consume the action (clears ActionReady flag)
                InputController.action = Instance.sharedMemory.ReadActionAndConsume();
                
                try {
                    // Execute exactly ONE update for this action
                    orig(self, gameTime);
                    updatesExecuted++;
                    // OnLevelUpdate writes state during orig()
                    
                } catch (Exception ex) {
                    Logger.Log(LogLevel.Error, "CelesteGym", 
                        $"Exception during update: {ex.Message}\n{ex.StackTrace}");
                    Settings.FastForwardEnabled = false;
                    break;
                }
            }
            // else: skip this slot, Python hasn't provided new action yet
        }
        
        Instance.skippedUpdates += (Settings.UpdatesPerFrame - updatesExecuted);
        
        if (Instance.renderFrameCount % 600 == 0) {
            float actualSpeedup = Instance.updateCount / (float)Instance.renderFrameCount;
            float skipRate = Instance.skippedUpdates / (float)(Instance.updateCount + Instance.skippedUpdates) * 100;
            
            Logger.Log(LogLevel.Info, "CelesteGym", 
                $"Performance: {actualSpeedup:F1}x speedup, Skip rate: {skipRate:F1}%");
            
            Instance.updateCount = 0;
            Instance.renderFrameCount = 0;
            Instance.skippedUpdates = 0;
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
        Instance.currentState.Facing = (byte)(player.Facing == Facings.Left ? 255 : 1);
        
        Instance.currentState.State = (byte)player.StateMachine.State;

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
    /// Get current performance metrics.
    /// </summary>
    public (uint updates, uint frames, float speedup) GetMetrics() {
        float speedup = renderFrameCount > 0 ? updateCount / (float)renderFrameCount : 0;
        return (updateCount, renderFrameCount, speedup);
    }
}