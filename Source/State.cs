using System;
using System.Text;
using System.Runtime.InteropServices;
using Celeste;
using Celeste.Mod;
using Monocle;
using Microsoft.Xna.Framework;

namespace Celeste.Mod.CelesteGym;

[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct GameState {
    // Player state (32 bytes)
    public float PosX;
    public float PosY;
    public float VelX;
    public float VelY;
    public float Stamina;
    public byte Dashes;
    public byte OnGround;
    public byte Transitioning;  
    public byte Facing;        // -1 or 1
    public byte State;         // Player state machine state
    public byte Dead;          // 0 or 1
    public byte CanDash;       // 0 or 1
    public byte Padding1;
    public uint FrameCount;
    
    // Local observation grid (1024 bytes)
    // 32x32 grid, each cell = 8x8 logical pixels
    public unsafe fixed byte LocalGrid[32 * 32];
}

public enum TileType : byte {
    Empty = 0,
    Solid = 1,
    Spike = 2,
    Platform = 3,
    Spring = 4,
    Refill = 5,
    Strawberry = 6,
    Goal = 7
}

public class GridManager {
    private const int GRID_SIZE = 32;
    private const int CELL_SIZE = 8;  // 8x8 logical pixels
    private const int HALF_GRID = GRID_SIZE / 2;
    
    public unsafe static void BuildLocalGrid(Level level, Player player, byte* gridPtr) {
        // Clear grid
        for (int i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
            gridPtr[i] = (byte)TileType.Empty;
        }
        
        float centerX = player.Center.X;
        float centerY = player.Center.Y;
        
        // Scan solids
        ScanSolids(level, centerX, centerY, gridPtr);
        
        // Scan hazards (spikes)
        ScanSpikes(level, centerX, centerY, gridPtr);
        
        // Scan platforms
        ScanPlatforms(level, centerX, centerY, gridPtr);
        
        // Scan entities (springs, refills, etc.)
        // ScanEntities(level, centerX, centerY, gridPtr);
    }
    public unsafe static string DumpGrid(byte* gridPtr) {
        StringBuilder sb = new StringBuilder();
        
        for (int y = 0; y < GRID_SIZE; y++) {
            for (int x = 0; x < GRID_SIZE; x++) {
                byte tile = gridPtr[y * GRID_SIZE + x];
                sb.Append(TileToChar(tile));
            }
            sb.AppendLine();
        }
        
        return sb.ToString();
    }

    private static char TileToChar(byte tileType){
        return (TileType) tileType switch {
            TileType.Empty => '.',
            TileType.Solid => '#',
            TileType.Spike => 'X',
            TileType.Platform => '=',
            TileType.Spring => 'S',
            TileType.Refill => 'R',
            TileType.Strawberry => 'B',  // Berry
            TileType.Goal => 'G',
            _ => '?'
        };
    }

    private unsafe static void ScanSolids(Level level, float centerX, float centerY, byte* gridPtr) {
        SolidTiles tiles = level.SolidTiles;
        Grid grid = tiles.Grid;
        
        for (int gy = 0; gy < GRID_SIZE; gy++) {
            for (int gx = 0; gx < GRID_SIZE; gx++) {
                // World position this grid cell represents (center of cell)
                float worldX = centerX + (gx - HALF_GRID) * CELL_SIZE;
                float worldY = centerY + (gy - HALF_GRID) * CELL_SIZE;
                
                // Convert to tile coordinates
                int tileX = (int)((worldX - grid.AbsoluteLeft) / CELL_SIZE);
                int tileY = (int)((worldY - grid.AbsoluteTop) / CELL_SIZE);
                
                // Check bounds and if solid
                if (tileX >= 0 && tileX < grid.CellsX && 
                    tileY >= 0 && tileY < grid.CellsY && 
                    grid[tileX, tileY]) {
                    gridPtr[gy * GRID_SIZE + gx] = (byte)TileType.Solid;
                }
            }
        }
    }
    
    private unsafe static void ScanSpikes(Level level, float centerX, float centerY, byte* gridPtr) {
        foreach (Spikes spike in level.Tracker.GetEntities<Spikes>()) {
            if (!spike.Collidable) continue;
            
            // Get spike bounds
            Rectangle bounds = spike.Collider.Bounds;
            
            // Convert spike center to grid coordinates
            float spikeX = bounds.Center.X;
            float spikeY = bounds.Center.Y;
            
            int gx = (int)((spikeX - centerX) / CELL_SIZE + HALF_GRID);
            int gy = (int)((spikeY - centerY) / CELL_SIZE + HALF_GRID);
            
            // Mark spike in grid (overrides solid)
            if (gx >= 0 && gx < GRID_SIZE && gy >= 0 && gy < GRID_SIZE) {
                gridPtr[gy * GRID_SIZE + gx] = (byte)TileType.Spike;
            }
        }
    }
    
    private unsafe static void ScanPlatforms(Level level, float centerX, float centerY, byte* gridPtr) {
        foreach (Platform platform in level.Tracker.GetEntities<Platform>()) {
            if (!platform.Collidable || platform is SolidTiles) continue;
            
            // Get platform bounds
            Rectangle bounds = platform.Collider.Bounds;
            
            // Mark all cells covered by platform
            int minGx = (int)((bounds.Left - centerX) / CELL_SIZE + HALF_GRID);
            int maxGx = (int)((bounds.Right - centerX) / CELL_SIZE + HALF_GRID);
            int minGy = (int)((bounds.Top - centerY) / CELL_SIZE + HALF_GRID);
            int maxGy = (int)((bounds.Bottom - centerY) / CELL_SIZE + HALF_GRID);
            
            for (int gy = Math.Max(0, minGy); gy <= Math.Min(GRID_SIZE - 1, maxGy); gy++) {
                for (int gx = Math.Max(0, minGx); gx <= Math.Min(GRID_SIZE - 1, maxGx); gx++) {
                    // Only mark if not already spike/solid
                    if (gridPtr[gy * GRID_SIZE + gx] == (byte)TileType.Empty) {
                        gridPtr[gy * GRID_SIZE + gx] = (byte)TileType.Platform;
                    }
                }
            }
        }
    }
    
    private unsafe static void ScanEntities(Level level, float centerX, float centerY, byte* gridPtr) {
        // Springs
        foreach (Spring spring in level.Tracker.GetEntities<Spring>()) {
            MarkEntity(spring, centerX, centerY, gridPtr, TileType.Spring);
        }
        
        // Refills
        foreach (Refill refill in level.Tracker.GetEntities<Refill>()) {
            MarkEntity(refill, centerX, centerY, gridPtr, TileType.Refill);
        }
        
        // Strawberries
        foreach (Strawberry berry in level.Tracker.GetEntities<Strawberry>()) {
            MarkEntity(berry, centerX, centerY, gridPtr, TileType.Strawberry);
        }
    }
    
    private unsafe static void MarkEntity(Entity entity, float centerX, float centerY, byte* gridPtr, TileType type) {
        if (!entity.Collidable) return;
        
        float entityX = entity.Center.X;
        float entityY = entity.Center.Y;
        
        int gx = (int)((entityX - centerX) / CELL_SIZE + HALF_GRID);
        int gy = (int)((entityY - centerY) / CELL_SIZE + HALF_GRID);
        
        if (gx >= 0 && gx < GRID_SIZE && gy >= 0 && gy < GRID_SIZE) {
            gridPtr[gy * GRID_SIZE + gx] = (byte)type;
        }
    }
}