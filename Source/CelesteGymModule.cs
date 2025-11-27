using System;
using Celeste;
using Celeste.Mod;
using Monocle;
using Microsoft.Xna.Framework;

namespace Celeste.Mod.CelesteGym;

public class CelesteGymModule : EverestModule {

    public static CelesteGymModule Instance { get; private set; } = null!;

    public override Type SettingsType => typeof(CelesteGymModuleSettings);
    public static CelesteGymModuleSettings Settings => (CelesteGymModuleSettings) Instance._Settings;

    public override Type SessionType => typeof(CelesteGymModuleSession);
    public static CelesteGymModuleSession Session => (CelesteGymModuleSession) Instance._Session;

    public override Type SaveDataType => typeof(CelesteGymModuleSaveData);
    public static CelesteGymModuleSaveData SaveData => (CelesteGymModuleSaveData) Instance._SaveData;

    private GameState currentState;
    private SharedMemoryBridge? sharedMemory;
    
    public CelesteGymModule() {
        Instance = this;
#if DEBUG
        // debug builds use verbose logging
        Logger.SetLogLevel(nameof(CelesteGymModule), LogLevel.Verbose);
#else
        // release builds use info logging to reduce spam in log files
        Logger.SetLogLevel(nameof(CelesteGymModule), LogLevel.Info);
#endif
    }

    public override void Load() {
        sharedMemory = new SharedMemoryBridge();
        if (!sharedMemory.Initialize()) {
            Logger.Log(LogLevel.Error, "CelesteGym", "Failed to initialize shared memory!");
            return;
        }
        Everest.Events.Level.OnBeforeUpdate += OnLevelUpdate;
        
        Logger.Log(LogLevel.Info, "CelesteGym", "Module loaded successfully");
    }
    public override void Unload() {
        Everest.Events.Level.OnBeforeUpdate -= OnLevelUpdate;
               
        sharedMemory?.Dispose();
        sharedMemory = null;
        
        Logger.Log(LogLevel.Info, "CelesteGym", "Module unloaded");
    }
    private void OnLevelUpdate(Level level) {
        if (sharedMemory == null) return;
        
        Player player = level.Tracker.GetEntity<Player>();
        if (player == null) return;
        
        // Extract state
        ExtractState(level);
        
        // Write to shared memory
        sharedMemory.WriteState(ref currentState);
        
        // Read action from Python
        ushort action = sharedMemory.ReadAction();
        Logger.Log(LogLevel.Info, "CelesteGym", $"Action: {action}");
        // Apply action (TODO: implement InputController)
        // InputController.ApplyAction(action);
    }
    private void ExtractState(Level level) {
        Player player = level.Tracker.GetEntity<Player>();
        if (player == null) return;
        
        // Fill player state
        currentState.PosX = player.Position.X;
        currentState.PosY = player.Position.Y;
        currentState.VelX = player.Speed.X;
        currentState.VelY = player.Speed.Y;
        currentState.Stamina = player.Stamina;
        currentState.Dashes = (byte)player.Dashes;
        currentState.OnGround = (byte)(player.OnGround() ? 1 : 0);
        currentState.Dead = (byte)(player.Dead ? 1 : 0);
        currentState.FrameCount++;
        
        // Build observation grid
        unsafe {
            fixed (byte* gridPtr = currentState.LocalGrid) {
                GridManager.BuildLocalGrid(level, player, gridPtr);
                if(currentState.FrameCount % 600 == 0){
                    string gridDump = GridManager.DumpGrid(gridPtr);
                    Logger.Log(LogLevel.Info, "CelesteGym", $"{currentState.FrameCount}");
                    Logger.Log(LogLevel.Info, "CelesteGym", "Grid state:\n" + gridDump);
                }
            }
        }
        
        // TODO: Write to shared memory
        //Logger.Log(LogLevel.Info, "CelesteGym", 
        //   $"State: Pos=({currentState.PosX:F1}, {currentState.PosY:F1}) " +
        //  $"Vel=({currentState.VelX:F1}, {currentState.VelY:F1}) " +
        //   $"Frame={currentState.FrameCount}");
    }
}
