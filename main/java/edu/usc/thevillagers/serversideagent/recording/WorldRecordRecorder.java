package edu.usc.thevillagers.serversideagent.recording;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import edu.usc.thevillagers.serversideagent.recording.event.RecordEvent;
import edu.usc.thevillagers.serversideagent.recording.event.RecordEventEntityDie;
import edu.usc.thevillagers.serversideagent.recording.event.RecordEventEntitySpawn;
import edu.usc.thevillagers.serversideagent.recording.event.RecordEventEntityUpdate;
import edu.usc.thevillagers.serversideagent.recording.event.RecordEventTileEntityDie;
import edu.usc.thevillagers.serversideagent.recording.event.RecordEventTileEntitySpawn;
import edu.usc.thevillagers.serversideagent.recording.event.RecordEventTileEntityUpdate;
import net.minecraft.entity.Entity;
import net.minecraft.entity.EntityList;
import net.minecraft.nbt.NBTTagCompound;
import net.minecraft.tileentity.TileEntity;
import net.minecraft.util.math.AxisAlignedBB;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.World;

public class WorldRecordRecorder extends WorldRecordWorker {
	
	public World world;
	private List<RecordEvent> currentTickEvents;
	
	public WorldRecordRecorder(World world, BlockPos from, BlockPos to) {
		this.world = world;
		this.from = from;
		this.to = to;
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
		if(currentTick % snapshotLenght == 0) newSnapshot();
		currentTickEvents = new ArrayList<>();
	}
	
	public void recordEvent(RecordEvent event) {
		if(currentTickEvents != null)
			currentTickEvents.add(event);
	}
	
	public Map<Integer, NBTTagCompound> computeEntitiesData(World world) {
		List<Entity> entities = world.<Entity>getEntitiesWithinAABB(Entity.class, new AxisAlignedBB(from, to));
		Map<Integer, NBTTagCompound> data = new HashMap<>();
		for(Entity e : entities)
			data.put(e.getEntityId(), e.writeToNBT(new NBTTagCompound()));
		return data;
	}
	
	public Map<BlockPos, NBTTagCompound> computeTileEntitiesData(World world) {
		//TODO optimise by only looking up TileEntities within relevent chunks
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
