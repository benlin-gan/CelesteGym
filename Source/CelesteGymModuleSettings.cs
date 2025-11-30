namespace Celeste.Mod.CelesteGym;

/// <summary>
/// Settings for CelesteGym module.
/// Accessible in-game via Mod Options menu.
/// </summary>
public class CelesteGymModuleSettings : EverestModuleSettings {
    
    [SettingName("Fast-Forward Enabled")]
    [SettingSubText("Enable fast-forward mode for RL training")]
    public bool FastForwardEnabled { get; set; } = true;
    
    [SettingName("Updates Per Frame")]
    [SettingSubText("Number of game updates per render frame (speedup factor)")]
    [SettingRange(1, 1000)]
    public int UpdatesPerFrame { get; set; } = 1;
    
    [SettingName("State Logging Interval")]
    [SettingSubText("Log game state every N frames (0 = disabled)")]
    [SettingRange(0, 10000)]
    public int StateLoggingInterval { get; set; } = 2400;
}