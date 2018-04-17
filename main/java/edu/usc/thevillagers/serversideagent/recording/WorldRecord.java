package edu.usc.thevillagers.serversideagent.recording;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import net.minecraft.nbt.CompressedStreamTools;
import net.minecraft.nbt.NBTTagCompound;
import net.minecraft.nbt.NBTUtil;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.IBlockAccess;

public class WorldRecord {
	
	private int snapshotLenght = 1 * 60 * 20; // one minute
	
	public IBlockAccess world;
	public BlockPos from, to;
	
	public final File saveFolder;
	
	private int duration;
	private int currentTick;
	
	private List<RecordEvent> currentTickEvents;
	private ChangeSet currentChangeSet;
	
	public WorldRecord(IBlockAccess world, BlockPos from, BlockPos to) {
		this.world = world;
		this.from = from;
		this.to = to;
		this.saveFolder = new File("tmp/rec");
		this.currentTick = 0;
	}
	
	public WorldRecord(File saveFolder) {
		this.world = null;
		this.saveFolder = saveFolder;
		this.currentTick = 0;
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
	
	public void endRecordTick() {
		if(currentTickEvents == null) return;
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
