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

public class WorldRecordWorker {

	protected int snapshotLenght = 1 * 60 * 20;
	public BlockPos from;
	public BlockPos to;
	public File saveFolder;
	public int duration;
	
	public int currentTick;
	protected ChangeSet currentChangeSet;
	
	public Map<Integer, NBTTagCompound> entitiesData;
	public Map<BlockPos, NBTTagCompound> tileEntitiesData;
	
	protected ExecutorService ioExecutor = Executors.newSingleThreadExecutor();
	
	public WorldRecordWorker() {
		this.currentTick = 0;
		entitiesData = new HashMap<>();
		tileEntitiesData = new HashMap<>();
	}
	
	public void writeInfo() throws IOException {
		File infoFile = new File(saveFolder, "record.info");
		infoFile.getParentFile().mkdirs();
		infoFile.createNewFile();
		NBTTagCompound comp = new NBTTagCompound();
		comp.setTag("From", NBTUtil.createPosTag(from));
		comp.setTag("To"  , NBTUtil.createPosTag(to  ));
		comp.setInteger("SnapshotLenght", snapshotLenght);
		comp.setInteger("Duration", duration);
		NBTFileInterface.writeToFile(comp, infoFile);
	}
	
	public void readInfo() throws IOException {
		File infoFile = new File(saveFolder, "record.info");
		NBTTagCompound comp = NBTFileInterface.readFromFile(infoFile);
		from = NBTUtil.getPosFromTag(comp.getCompoundTag("From"));
		to   = NBTUtil.getPosFromTag(comp.getCompoundTag("To"  ));
		snapshotLenght = comp.getInteger("SnapshotLenght");
		duration = comp.getInteger("Duration");
	}
	
	protected ChangeSet changeSet(int tick) {
		return new ChangeSet(new File(saveFolder, tick / snapshotLenght + ".changeset"));
	}

	protected Snapshot snapshot(int tick) {
		return new Snapshot(new File(saveFolder, tick / snapshotLenght + ".snapshot"));
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