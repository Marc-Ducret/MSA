package edu.usc.thevillagers.serversideagent.recording;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import net.minecraft.nbt.NBTTagCompound;
import net.minecraft.nbt.NBTUtil;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;

/**
 * A worker that can interact with a record. Can be extended to record or replay.
 */
public class WorldRecordWorker {

	protected int snapshotLength = 1 * 60 * 20;
	public BlockPos from;
	public BlockPos to;
	public File saveFolder;
	public int duration;
	
	public int currentTick;
	protected ChangeSet currentChangeSet;
	
	private Map<Integer, Snapshot> snapshotCache;
	private Map<Integer, ChangeSet> changeSetCache;
	
	public Map<Integer, NBTTagCompound> entitiesData;
	public Map<BlockPos, NBTTagCompound> tileEntitiesData;
	
	public World world;
	
	protected ExecutorService ioExecutor = Executors.newSingleThreadExecutor();
	
	public WorldRecordWorker() {
		this.currentTick = 0;
		entitiesData = new HashMap<>();
		tileEntitiesData = new HashMap<>();
		snapshotCache = new HashMap<>();
		changeSetCache = new HashMap<>();
	}
	
	public void writeInfo() throws IOException {
		File infoFile = new File(saveFolder, "record.info");
		infoFile.getParentFile().mkdirs();
		infoFile.createNewFile();
		NBTTagCompound comp = new NBTTagCompound();
		comp.setTag("From", NBTUtil.createPosTag(from));
		comp.setTag("To"  , NBTUtil.createPosTag(to  ));
		comp.setInteger("SnapshotLenght", snapshotLength);
		comp.setInteger("Duration", duration);
		NBTFileInterface.writeToFile(comp, infoFile);
	}
	
	public void readInfo() throws IOException {
		File infoFile = new File(saveFolder, "record.info");
		NBTTagCompound comp = NBTFileInterface.readFromFile(infoFile);
		from = NBTUtil.getPosFromTag(comp.getCompoundTag("From"));
		to   = NBTUtil.getPosFromTag(comp.getCompoundTag("To"  ));
		snapshotLength = comp.getInteger("SnapshotLenght");
		duration = comp.getInteger("Duration");
	}
	
	protected ChangeSet changeSet(int tick) {
		int id = tick / snapshotLength;
		if(!changeSetCache.containsKey(id)) changeSetCache.put(id, new ChangeSet(new File(saveFolder, id + ".changeset")));
		return changeSetCache.get(id);
	}

	protected Snapshot snapshot(int tick) {
		int id = tick / snapshotLength;
		if(!snapshotCache.containsKey(id)) snapshotCache.put(id, new Snapshot(new File(saveFolder, tick / snapshotLength + ".snapshot")));
		return snapshotCache.get(id);
	}

	public static NBTTagCompound computeDifferentialCompound(NBTTagCompound oldComp, NBTTagCompound newComp) {
		NBTTagCompound diffComp = newComp.copy();
		for(String key : newComp.getKeySet())
			if(newComp.getTag(key).equals(oldComp.getTag(key)))
				diffComp.removeTag(key);
		return diffComp;
	}

	public static void updateCompound(NBTTagCompound oldComp, NBTTagCompound updateComp) {
		for(String key : updateComp.getKeySet())
			oldComp.setTag(key, updateComp.getTag(key));
	}
}