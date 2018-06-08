package edu.usc.thevillagers.serversideagent.recording;

import java.io.File;
import java.io.IOException;
import java.util.UUID;
import java.util.concurrent.ExecutionException;

import com.mojang.authlib.GameProfile;

import edu.usc.thevillagers.serversideagent.ServerSideAgentMod;
import it.unimi.dsi.fastutil.longs.Long2ObjectMap;
import net.minecraft.client.Minecraft;
import net.minecraft.client.entity.EntityPlayerSP;
import net.minecraft.client.multiplayer.ChunkProviderClient;
import net.minecraft.client.multiplayer.PlayerControllerMP;
import net.minecraft.client.multiplayer.WorldClient;
import net.minecraft.client.network.NetHandlerPlayClient;
import net.minecraft.entity.Entity;
import net.minecraft.network.EnumPacketDirection;
import net.minecraft.network.NetworkManager;
import net.minecraft.stats.RecipeBook;
import net.minecraft.stats.StatisticsManager;
import net.minecraft.util.MovementInput;
import net.minecraft.world.EnumDifficulty;
import net.minecraft.world.GameType;
import net.minecraft.world.World;
import net.minecraft.world.WorldSettings;
import net.minecraft.world.WorldType;
import net.minecraft.world.chunk.Chunk;
import net.minecraftforge.fml.relauncher.Side;
import net.minecraftforge.fml.relauncher.SideOnly;

/**
 * {@link WorldRecordReplayer} with rendering capabilities.
 */
@SideOnly(value=Side.CLIENT)
public class WorldRecordReplayerClient extends WorldRecordReplayer {
	
	public WorldClient world;
	public EntityPlayerSP player;
	public PlayerControllerMP playerController;

	public WorldRecordReplayerClient(File saveFolder) {
		super(saveFolder);
	}
	
	@Override
	protected World createWorld() {
		WorldSettings settings = new WorldSettings(0, GameType.SPECTATOR, false, false, WorldType.FLAT);
		GameProfile profile = new GameProfile(UUID.randomUUID(), "dummy");
		Minecraft mc = Minecraft.getMinecraft();
		NetHandlerPlayClient nethandler = new NetHandlerPlayClient(mc, mc.currentScreen, new NetworkManager(EnumPacketDirection.CLIENTBOUND), profile);
		
		world = new WorldClient(nethandler, settings, 0, EnumDifficulty.PEACEFUL, mc.mcProfiler) {
			@Override
			public Entity getEntityByID(int id) {
				return this.entitiesById.lookup(id);
			}
		};
		if(player != null) {
			player.setWorld(world);
		} else {
			player = new EntityPlayerSP(mc, world, nethandler, new StatisticsManager(), new RecipeBook()) {
				
				@Override
				public float getCooledAttackStrength(float adjustTicks) {
					return 1F;
				}
				
				@Override
				public boolean isSpectator() {
					return true;
				}
			};
			player.setGameType(GameType.SPECTATOR);
			player.movementInput = new MovementInput();
			playerController = new PlayerControllerMP(mc, nethandler);
		}
		return world;
	}
	
	@Override
	public void seek(int tick) throws IOException, InterruptedException, ExecutionException {
		super.seek(tick);
		Minecraft.getMinecraft().renderGlobal.setWorldAndLoadRenderers(world);
	}
	
	@Override
	public Long2ObjectMap<Chunk> getChunkMapping() {
		return ServerSideAgentMod.getPrivateField(ChunkProviderClient.class, "chunkMapping", world.getChunkProvider());
	}
	
	@Override
	public void spawnEntity(Entity e) {
		e.forceSpawn = true;
		world.addEntityToWorld(e.getEntityId(), e);
	}
	
	@Override
	public void killEntity(int id) {
		super.killEntity(id);
		world.removeEntityFromWorld(id);
	}
}
