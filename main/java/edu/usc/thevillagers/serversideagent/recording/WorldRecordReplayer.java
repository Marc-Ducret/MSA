package edu.usc.thevillagers.serversideagent.recording;

import java.io.File;
import java.io.IOException;
import java.util.Map.Entry;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import edu.usc.thevillagers.serversideagent.recording.event.RecordEvent;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.nbt.NBTTagCompound;
import net.minecraft.server.MinecraftServer;
import net.minecraft.tileentity.TileEntity;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.GameType;
import net.minecraft.world.World;
import net.minecraft.world.WorldServer;
import net.minecraft.world.WorldSettings;
import net.minecraft.world.WorldType;
import net.minecraft.world.storage.WorldInfo;
import net.minecraftforge.common.DimensionManager;
import net.minecraftforge.fml.common.FMLCommonHandler;

/**
 * Replays a world record from the disk.
 */
public class WorldRecordReplayer extends WorldRecordWorker {
	
	public static final int DUMMY_DIMENSION = 0xDEAD;
	
	public long worldTimeOffset;
	
	private Future<ChangeSet> nextChangeSet;
	
	public WorldRecordReplayer(File saveFolder) {
		this.world = null;
		this.saveFolder = saveFolder;
	}
	
	@Override
	public void readInfo() throws IOException {
		super.readInfo();
		world = createWorld();
	}
	
	protected World createWorld() {
		WorldSettings settings = new WorldSettings(0, GameType.NOT_SET, false, false, WorldType.FLAT);
		WorldInfo info = new WorldInfo(settings, "dummy");
		MinecraftServer server = FMLCommonHandler.instance().getMinecraftServerInstance();
		WorldServer world = new WorldServer(server, server.getActiveAnvilConverter().getSaveLoader("dummy", false),
				info, DUMMY_DIMENSION, server.profiler); //TODO custom ISaveHandler
		DimensionManager.setWorld(DUMMY_DIMENSION, null, server);
		return world;
	}
	
	public void endReplayTick() throws InterruptedException, ExecutionException {
		if(currentTick >= duration) return;
		int phase = currentTick % snapshotLength;
		if(phase == 0) {
			currentChangeSet = nextChangeSet.get();
			if(currentTick + snapshotLength < duration) {
				ChangeSet changeSet = changeSet(currentTick + snapshotLength);
				nextChangeSet = ioExecutor.submit(() -> {
					changeSet.read();
					return changeSet;
				});
			}
		}
		for(RecordEvent event : currentChangeSet.data.get(phase))
			event.replay(this);
		for(Entry<Integer, NBTTagCompound> entry : entitiesData.entrySet())
			updateEntity(entry.getKey(), entry.getValue());
		for(Entry<BlockPos, NBTTagCompound> entry : tileEntitiesData.entrySet())
			updateTileEntity(entry.getKey(), entry.getValue());
		currentTick++;
//		world.fakeWorld.setTotalWorldTime(worldTimeOffset + currentTick); TODO that's client side only...
		world.setWorldTime((worldTimeOffset + currentTick) % 24000);
	}
	
	public void seek(int tick) throws IOException, InterruptedException, ExecutionException {
		long start = System.currentTimeMillis();
		
		if(tick >= duration) tick = duration - 1;
		currentTick = tick - (tick % snapshotLength);
		
		Snapshot snapshot = snapshot(tick);
		snapshot.read(); 
		
		ChangeSet changeSet = changeSet(currentTick);
		nextChangeSet = ioExecutor.submit(() -> {
			changeSet.read();
			return changeSet;
		});
		
		long ioEnd = System.currentTimeMillis();
		
		snapshot.applyDataToWorld(this);
		long appSnapEnd = System.currentTimeMillis();
		
		currentChangeSet = null;
		while(currentTick < tick)
			endReplayTick();
		
		long end = System.currentTimeMillis();
		System.out.println(String.format("SEEK(%d) in %d ms [io = %d ms | snap = %d ms | changes = %d ms]", tick, end - start, ioEnd - start, appSnapEnd - ioEnd, end - appSnapEnd));
	}
	
	public void spawnEntity(Entity e) {
		world.spawnEntity(e);
	}

	public void killEntity(int id) {
		Entity e = world.getEntityByID(id);
		if(e == null) {
			System.out.println("Missing entity "+id);
			return;
		}
		world.removeEntity(e);
	}

	public void updateEntity(int id, NBTTagCompound data) {
		Entity e = world.getEntityByID(id);
		if(e == null) {
			System.out.println("Missing entity "+id);
			return;
		}
		e.readFromNBT(data);
		if(e instanceof EntityPlayer) {
			e.setSneaking(data.getBoolean("Sneaking"));
		}
	}

	public Entity getEntity(int entityId) {
		return world.getEntityByID(entityId);
	}

	public void spawnTileEntity(TileEntity tileEntity) {
		world.setTileEntity(tileEntity.getPos(), tileEntity);
	}

	public void killTileEntity(BlockPos pos) {
		world.setTileEntity(pos, null);
	}

	public void updateTileEntity(BlockPos pos, NBTTagCompound data) {
		TileEntity tileEntity = world.getTileEntity(pos);
		if(tileEntity == null) {
			System.out.println("Missing tile entity at "+pos);
			return;
		}
		tileEntity.readFromNBT(data);
	}

	public void reset() { //TODO do
//		Arrays.fill(blockBuffer, Blocks.AIR.getDefaultState());
//		entities.clear();
//		tileEntities.clear();
	}
}
