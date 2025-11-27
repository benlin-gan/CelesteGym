using System;
using Celeste;
using Celeste.Mod;
using Monocle;
using Microsoft.Xna.Framework;

namespace Celeste.Mod.CelesteGym;

public class CelesteGymModule : EverestModule {
    public static CelesteGymModule Instance { get; private set; }

    public override Type SettingsType => typeof(CelesteGymModuleSettings);
    public static CelesteGymModuleSettings Settings => (CelesteGymModuleSettings) Instance._Settings;

    public override Type SessionType => typeof(CelesteGymModuleSession);
    public static CelesteGymModuleSession Session => (CelesteGymModuleSession) Instance._Session;

    public override Type SaveDataType => typeof(CelesteGymModuleSaveData);
    public static CelesteGymModuleSaveData SaveData => (CelesteGymModuleSaveData) Instance._SaveData;

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
        Everest.Events.Level.OnBeforeUpdate += TestHook;
    }

    public override void Unload() {
        Everest.Events.Level.OnBeforeUpdate -= TestHook;
    }
    private static void TestHook(Level level) {
        Player player = level.Tracker.GetEntity<Player>();
        if(player != null){
            Logger.Log(LogLevel.Info, "YourModName", SerializePlayer(player));
        }
    }
    private static string SerializePlayer(Player player){
        return $"{player.Position.X:F1}, {player.Position.Y:F1}";
    }
}
