package edu.usc.thevillagers.serversideagent.recording;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import edu.usc.thevillagers.serversideagent.recording.event.RecordEvent;
import edu.usc.thevillagers.serversideagent.recording.event.RecordEventEntityDie;
import edu.usc.thevillagers.serversideagent.recording.event.RecordEventEntitySpawn;
import edu.usc.thevillagers.serversideagent.recording.event.RecordEventEntityUpdate;
import net.minecraft.entity.Entity;
import net.minecraft.entity.EntityList;
import net.minecraft.nbt.CompressedStreamTools;
import net.minecraft.nbt.NBTTagCompound;
import net.minecraft.nbt.NBTUtil;
import net.minecraft.util.math.AxisAlignedBB;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.IBlockAccess;
import net.minecraft.world.World;

public class WorldRecord {
	
	private int snapshotLenght = 5 * 60 * 20; // five minute
	
	public IBlockAccess world;
	public BlockPos from, to;
	
	public final File saveFolder;
	
	private int duration;
	private int currentTick;
	
	private List<RecordEvent> currentTickEvents;
	private ChangeSet currentChangeSet;
	
	public Map<Integer, NBTTagCompound> entitiesData;
	
	public WorldRecord(IBlockAccess world, BlockPos from, BlockPos to) {
		this.world = world;
		this.from = from;
		this.to = to;
		this.saveFolder = new File("tmp/rec");
		this.currentTick = 0;
		entitiesData = new HashMap<>();
	}
	
	public WorldRecord(File saveFolder) {
		this.world = null;
		this.saveFolder = saveFolder;
		this.currentTick = 0;
		entitiesData = new HashMap<>();
	}
	
	public World getRecordWorld() {
		return (World) world;
	}
	
	public ReplayWorldAccess getReplayWorld() {
		return (ReplayWorldAccess) world;
	}
	
	public void writeInfo() throws IOException {
		File infoFile = new File(saveFolder, "record.info");
		infoFile.getParentFile().mkdirs();
		NBTTagCompound comp = new NBTTagCompound();
		comp.setTag("From", NBTUtil.createPosTag(from));
		comp.setTag("To"  , NBTUtil.createPosTag(to  ));
		comp.setInteger("SnapshotLenght", snapshotLenght);
		comp.setInteger("Duration", duration);
		CompressedStreamTools.write(comp, infoFile);
	}
	
	public void readInfo() throws IOException {
		File infoFile = new File(saveFolder, "record.info");
		NBTTagCompound comp = CompressedStreamTools.read(infoFile);
		from = NBTUtil.getPosFromTag(comp.getCompoundTag("From"));
		to   = NBTUtil.getPosFromTag(comp.getCompoundTag("To"  ));
		snapshotLenght = comp.getInteger("SnapshotLenght");
		duration = comp.getInteger("Duration");
		world = new ReplayWorldAccess(from, to);
	}
	
	public void startRecordTick() throws IOException {
		if(currentTick % snapshotLenght == 0) {
			writeInfo();
			Snapshot snapshot = new Snapshot(new File(saveFolder, currentTick / snapshotLenght + ".snapshot"));
			snapshot.setDataFromWorld(world, from, to);
			snapshot.write();
			if(currentChangeSet != null)
				currentChangeSet.write();
			currentChangeSet = new ChangeSet(new File(saveFolder, currentTick / snapshotLenght + ".changeset"));
			currentChangeSet.resetEventList();
		}
		currentTickEvents = new ArrayList<>();
	}
	
	public void recordEvent(RecordEvent event) {
		currentTickEvents.add(event);
	}
	
	private void recordEntities() {
		World world = (World) this.world;
		List<Entity> entities = world.<Entity>getEntitiesWithinAABB(Entity.class, new AxisAlignedBB(from, to));
		Map<Integer, NBTTagCompound> newData = new HashMap<>();
		for(Entity e : entities)
			newData.put(e.getEntityId(), e.writeToNBT(new NBTTagCompound()));
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
	
	public void endRecordTick() {
		if(currentTickEvents == null) return;
		recordEntities();
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
	
	public void endReplayTick() {
		ReplayWorldAccess world = (ReplayWorldAccess) this.world;
		for(Entry<Integer, NBTTagCompound> entry : entitiesData.entrySet())
			world.updateEntity(entry.getKey(), entry.getValue());
	}
	
	public static NBTTagCompound computeDifferentialCompound(NBTTagCompound oldComp, NBTTagCompound newComp) {
		NBTTagCompound diffComp = newComp.copy();
		for(String key : newComp.getKeySet())
			if(oldComp.getTag(key).equals(newComp.getTag(key)))
				diffComp.removeTag(key);
		return diffComp;
	}
	
	public static void updateCompound(NBTTagCompound oldComp, NBTTagCompound updateComp) {
		for(String key : updateComp.getKeySet())
			oldComp.setTag(key, updateComp.getTag(key));
	}
}
