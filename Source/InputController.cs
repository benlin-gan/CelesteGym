using System.Reflection;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Input;
using Monocle;

namespace Celeste.Mod.CelesteGym;

public static class InputController {
    public static ushort action = 0;
    
    static InputController() {
       
    }
    
    /// <summary>
    /// Apply action to game input.
    /// Action encoding (bitfield):
    ///   bit 0: Left
    ///   bit 1: Right  
    ///   bit 2: Jump (A button)
    ///   bit 3: Dash (X button)
    ///   bit 4: Grab (Right Trigger)
    /// </summary>
    public static void ApplyAction() {
        // Decode action bits
        Logger.Log(LogLevel.Info, "CelesteGym", $"ApplyAction called: {action}");
        bool left = (action & 0x01) != 0;
        bool right = (action & 0x02) != 0;
        bool up = (action & 0x04) != 0;
        bool down = (action & 0x08) != 0;        
        bool jump = (action & 0x10) != 0;
        bool dash = (action & 0x20) != 0;
        bool grab = (action & 0x40) != 0;
        
        // Find or create active gamepad
        MInput.GamePadData activePad = GetOrCreateGamePad();
        
        // Build thumbstick state (movement)
        float x = left ? -1.0f : (right ? 1.0f : 0.0f);
        float y = 0.0f; // No up/down on left stick in Celeste
        var sticks = new GamePadThumbSticks(
            new Vector2(x, y),  // Left stick
            Vector2.Zero        // Right stick (unused)
        );
        
        // Build button state
        Buttons buttons = 0;
        if (jump) buttons |= Buttons.A;
        if (dash) buttons |= Buttons.X;
        
        var buttonState = new GamePadButtons(buttons);
        
        // Build trigger state (grab uses right trigger)
        var triggers = new GamePadTriggers(0.0f, grab ? 1.0f : 0.0f);
        
        // Create new gamepad state
        var newState = new GamePadState(
            sticks, 
            triggers, 
            buttonState, 
            new GamePadDPad()  // D-pad unused
        );
        
        // Inject the state
        activePad.PreviousState = activePad.CurrentState;
        activePad.CurrentState = newState;
    }
    
    public static MInput.GamePadData GetOrCreateGamePad() {
        // Find first attached gamepad
        for (int i = 0; i < 4; i++) {
            if (MInput.GamePads[i].Attached) {
                return MInput.GamePads[i];
            }
        }
        
        // None found, create virtual gamepad
        MInput.GamePads[0].Attached = true;
        return MInput.GamePads[0];
    }
}