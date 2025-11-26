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
        On.Monocle.Engine.Update += TestHook;
    }

    public override void Unload() {
        On.Monocle.Engine.Update -= TestHook;
    }
    private static void TestHook(On.Monocle.Engine.orig_Update orig, Engine self, GameTime gameTime) {
        Logger.Log(LogLevel.Info, "YourModName", "Hook is working!");
        orig(self, gameTime);  // Don't forget to call the original!
    }
}
