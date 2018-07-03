package edu.usc.thevillagers.serversideagent.recording;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import com.mojang.authlib.GameProfile;

import edu.usc.thevillagers.serversideagent.ServerSideAgentMod;
import edu.usc.thevillagers.serversideagent.agent.Actor;
import edu.usc.thevillagers.serversideagent.recording.event.RecordEvent;
import edu.usc.thevillagers.serversideagent.recording.event.RecordEventEntityDie;
import edu.usc.thevillagers.serversideagent.recording.event.RecordEventEntitySpawn;
import edu.usc.thevillagers.serversideagent.recording.event.RecordEventEntityUpdate;
import edu.usc.thevillagers.serversideagent.recording.event.RecordEventTileEntityDie;
import edu.usc.thevillagers.serversideagent.recording.event.RecordEventTileEntitySpawn;
import edu.usc.thevillagers.serversideagent.recording.event.RecordEventTileEntityUpdate;
import net.minecraft.entity.Entity;
import net.minecraft.entity.EntityList;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.entity.player.EntityPlayerMP;
import net.minecraft.nbt.NBTTagCompound;
import net.minecraft.tileentity.TileEntity;
import net.minecraft.util.math.AxisAlignedBB;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.World;

/**
 * {@link WorldRecordWorker} for recording the world during gameplay.
 */
public class WorldRecordRecorder extends WorldRecordWorker {
	
	private List<RecordEvent> currentTickEvents;
	private AxisAlignedBB bounds;
	
	public WorldRecordRecorder(World world, BlockPos from, BlockPos to) {
		this.world = world;
		this.from = from;
		this.to = to;
		this.bounds = new AxisAlignedBB(from, to);
		String name = String.format("%1$ty_%1$tm_%1$td-%1$tH_%1$tM_%1$tS", Calendar.getInstance());
		this.saveFolder = new File("tmp/records/"+name);
	}
	
	public void startRecord() {
		entitiesData = computeEntitiesData(world);
		tileEntitiesData = computeTileEntitiesData(world);
	}
	
	private void newSnapshot() throws IOException {
		writeInfo();
		Snapshot snapshot = snapshot(currentTick);
		snapshot.setDataFromWorld(this);
		ioExecutor.execute(() -> {
			try {
				snapshot.write();
			} catch (IOException e) {
				System.err.println("Cannot write snapshot");
				e.printStackTrace();
			}
		});
		if(currentChangeSet != null) {
			ChangeSet changeSet = currentChangeSet;
			ioExecutor.execute(() -> {
				try {
					changeSet.write();
				} catch (IOException e) {
					System.err.println("Cannot write change set");
					e.printStackTrace();
				}
			});
		}
		currentChangeSet = changeSet(currentTick);
		currentChangeSet.resetEventList();
	}
	
	public void startRecordTick() throws IOException {
		if(currentTick % snapshotLength == 0) newSnapshot();
		currentTickEvents = new ArrayList<>();
	}
	
	public void recordEvent(RecordEvent event) {
		if(currentTickEvents != null && event.isWithinBounds(this, bounds))
			currentTickEvents.add(event);
	}
	
	private NBTTagCompound entityNBT(Entity e) {
		NBTTagCompound data = e.writeToNBT(new NBTTagCompound());
		if(e instanceof EntityPlayer) {
			data.setBoolean("Sneaking", e.isSneaking());
			GameProfile profile = ((EntityPlayer) e).getGameProfile();
			data.setString("ProfileName", profile.getName());
			data.setUniqueId("ProfileUUID", profile.getId());
			appendEnvInfo((EntityPlayerMP) e, data);
		}
		return data;
	}
	
	private void appendEnvInfo(EntityPlayerMP player, NBTTagCompound data) {
		Actor a = ServerSideAgentMod.instance.envManager.getPlayerActor(player);
		if(a != null) {
			data.setFloat("Reward", a.reward);
			data.setBoolean("Done", a.env.done);
		}
	}
	
	public Map<Integer, NBTTagCompound> computeEntitiesData(World world) {
		List<Entity> entities = world.<Entity>getEntitiesWithinAABB(Entity.class, bounds, (e) ->
			e != null && (!(e instanceof EntityPlayer) || !((EntityPlayer)e).isSpectator() || ((EntityPlayer)e).isCreative())
		);
		Map<Integer, NBTTagCompound> data = new HashMap<>();
		for(Entity e : entities)
			data.put(e.getEntityId(), entityNBT(e));
		return data;
	}
	
	public Map<BlockPos, NBTTagCompound> computeTileEntitiesData(World world) {
		//TODO optimise by only looking up TileEntities within relevant chunks
		Map<BlockPos, NBTTagCompound> data = new HashMap<>();
		AxisAlignedBB bounds = new AxisAlignedBB(from, to);
		for(TileEntity tileEntity : world.loadedTileEntityList)
			if(bounds.contains(new Vec3d(tileEntity.getPos())))
				data.put(tileEntity.getPos(), tileEntity.writeToNBT(new NBTTagCompound()));
		return data;
	}
	
	private void recordEntities() {
		Map<Integer, NBTTagCompound> newData = computeEntitiesData(world);
		for(Entry<Integer, NBTTagCompound> entry : newData.entrySet()) {
			if(!entitiesData.containsKey(entry.getKey())) {
				int type = EntityList.getID(world.getEntityByID(entry.getKey()).getClass());
				recordEvent(new RecordEventEntitySpawn(entry.getKey(), type, entry.getValue()));
			} else {
				NBTTagCompound diffData = computeDifferentialCompound(entitiesData.get(entry.getKey()), entry.getValue());
				if(diffData.getKeySet().size() > 0)
					recordEvent(new RecordEventEntityUpdate(entry.getKey(), diffData));
			}
		}
		for(int id : entitiesData.keySet())
			if(!newData.containsKey(id))
				recordEvent(new RecordEventEntityDie(id));
		
		entitiesData = newData;
	}
	
	private void recordTileEntities() {
		Map<BlockPos, NBTTagCompound> newData = computeTileEntitiesData(world);
		for(Entry<BlockPos, NBTTagCompound> entry : newData.entrySet()) {
			if(		!tileEntitiesData.containsKey(entry.getKey()) || 
					!tileEntitiesData.get(entry.getKey()).getString("id").equals(entry.getValue().getString("id"))) {
				recordEvent(new RecordEventTileEntitySpawn(entry.getKey(), entry.getValue()));
			} else {
				NBTTagCompound diffData = computeDifferentialCompound(tileEntitiesData.get(entry.getKey()), entry.getValue());
				if(diffData.getKeySet().size() > 0)
					recordEvent(new RecordEventTileEntityUpdate(entry.getKey(), diffData));
			}
		}
		for(BlockPos pos : tileEntitiesData.keySet())
			if(!newData.containsKey(pos))
				recordEvent(new RecordEventTileEntityDie(pos));
		
		tileEntitiesData = newData;
	}
	
	public void endRecordTick() {
		if(currentTickEvents == null) return;
		recordEntities();
		recordTileEntities();
		currentChangeSet.appendChanges(currentTickEvents);
		currentTick++;
		duration = currentTick;
	}
	
	public void endRecord() throws IOException {
		writeInfo();
		if(currentChangeSet != null)
			currentChangeSet.write();
		System.out.println("Recording completed ("+(duration / 20)+" seconds)");
	}
}
