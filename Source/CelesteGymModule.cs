using System;

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
        // TODO: apply any hooks that should always be active
    }

    public override void Unload() {
        // TODO: unapply any hooks applied in Load()
    }
}