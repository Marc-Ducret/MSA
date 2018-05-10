package edu.usc.thevillagers.serversideagent.recording;

import java.io.File;
import java.io.IOException;
import java.util.UUID;
import java.util.concurrent.ExecutionException;

import com.mojang.authlib.GameProfile;

import net.minecraft.client.Minecraft;
import net.minecraft.client.entity.EntityPlayerSP;
import net.minecraft.client.multiplayer.PlayerControllerMP;
import net.minecraft.client.multiplayer.WorldClient;
import net.minecraft.client.network.NetHandlerPlayClient;
import net.minecraft.entity.Entity;
import net.minecraft.network.EnumPacketDirection;
import net.minecraft.network.NetworkManager;
import net.minecraft.stats.RecipeBook;
import net.minecraft.stats.StatisticsManager;
import net.minecraft.world.EnumDifficulty;
import net.minecraft.world.GameType;
import net.minecraft.world.World;
import net.minecraft.world.WorldSettings;
import net.minecraft.world.WorldType;
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
		WorldSettings settings = new WorldSettings(0, GameType.NOT_SET, false, false, WorldType.FLAT);
		GameProfile profile = new GameProfile(UUID.randomUUID(), "dummy");
		Minecraft mc = Minecraft.getMinecraft();
		NetHandlerPlayClient nethandler = new NetHandlerPlayClient(mc, mc.currentScreen, new NetworkManager(EnumPacketDirection.CLIENTBOUND), profile);
		
		world = new WorldClient(nethandler, settings, 0, EnumDifficulty.PEACEFUL, mc.mcProfiler) {
			@Override
			public Entity getEntityByID(int id) {
				return this.entitiesById.lookup(id);
			}
		};
		player = new EntityPlayerSP(mc, world, nethandler, new StatisticsManager(), new RecipeBook());
		playerController = new PlayerControllerMP(mc, nethandler);
		return world;
	}
	
//	@Override
//	protected ReplayWorldAccess createWorld() {
//		return world = new ReplayWorldAccessClient(from, to);
//	}

	@Override
	public void seek(int tick) throws IOException, InterruptedException, ExecutionException {
		super.seek(tick);
//		for(int chunkZ = from.getZ() >> 4; chunkZ <= to.getZ() >> 4; chunkZ++)
//			for(int chunkY = from.getY() >> 4; chunkY <= to.getY() >> 4; chunkY++)
//				for(int chunkX = from.getX() >> 4; chunkX <= to.getX() >> 4; chunkX++)
//					world.chunkBufferManager.requestUpdate(chunkX, chunkY, chunkZ);
	}
	
	@Override
	public void spawnEntity(Entity e) {
		world.addEntityToWorld(e.getEntityId(), e);
	}
}
