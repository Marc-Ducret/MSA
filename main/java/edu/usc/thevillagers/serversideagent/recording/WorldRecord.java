package edu.usc.thevillagers.serversideagent.recording;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import edu.usc.thevillagers.serversideagent.recording.event.RecordEvent;
import edu.usc.thevillagers.serversideagent.recording.event.RecordEventEntityDie;
import edu.usc.thevillagers.serversideagent.recording.event.RecordEventEntitySpawn;
import edu.usc.thevillagers.serversideagent.recording.event.RecordEventEntityUpdate;
import net.minecraft.entity.Entity;
import net.minecraft.entity.EntityList;
import net.minecraft.nbt.NBTTagCompound;
import net.minecraft.nbt.NBTUtil;
import net.minecraft.util.math.AxisAlignedBB;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.IBlockAccess;
import net.minecraft.world.World;

public class WorldRecord {
	
	private int snapshotLenght = 1 * 20; // one minute
	
	public IBlockAccess world;
	public BlockPos from, to;
	
	public final File saveFolder;
	
	public int duration;
	public int currentTick;
	
	private List<RecordEvent> currentTickEvents;
	private ChangeSet currentChangeSet;
	
	public Map<Integer, NBTTagCompound> entitiesData;
	
	private Future<ChangeSet> nextChangeSet;
	
	private ExecutorService ioExecutor = Executors.newSingleThreadExecutor();
	
	public WorldRecord(IBlockAccess world, BlockPos from, BlockPos to) {
		this.world = world;
		this.from = from;
		this.to = to;
		String name = String.format("%1$ty_%1$tm_%1$td-%1$tH_%1$tM_%1$tS", Calendar.getInstance());
		this.saveFolder = new File("tmp/records/"+name);
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
		NBTFileInterface.writeToFile(comp, infoFile);
	}
	
	public void readInfo() throws IOException {
		File infoFile = new File(saveFolder, "record.info");
		NBTTagCompound comp = NBTFileInterface.readFromFile(infoFile);
		from = NBTUtil.getPosFromTag(comp.getCompoundTag("From"));
		to   = NBTUtil.getPosFromTag(comp.getCompoundTag("To"  ));
		snapshotLenght = comp.getInteger("SnapshotLenght");
		duration = comp.getInteger("Duration");
		world = new ReplayWorldAccess(from, to);
	}
	
	private ChangeSet changeSet(int tick) {
		return new ChangeSet(new File(saveFolder, tick / snapshotLenght + ".changeset"));
	}
	
	private Snapshot snapshot(int tick) {
		return new Snapshot(new File(saveFolder, tick / snapshotLenght + ".snapshot"));
	}
	
	public void startRecord() {
		entitiesData = computeEntitiesData((World) world);
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
	
	private void recordEntities() {
		World world = (World) this.world;
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
	
	public void endReplayTick() throws InterruptedException, ExecutionException {
		if(currentTick >= duration) return;
		int phase = currentTick % snapshotLenght;
		if(phase == 0) {
			currentChangeSet = nextChangeSet.get();
			if(currentTick + snapshotLenght < duration) {
				ChangeSet changeSet = changeSet(currentTick + snapshotLenght);
				nextChangeSet = ioExecutor.submit(() -> {
					changeSet.read();
					return changeSet;
				});
			}
		}
		for(RecordEvent event : currentChangeSet.data.get(phase))
			event.replay(this);
		ReplayWorldAccess world = getReplayWorld();
		for(Entry<Integer, NBTTagCompound> entry : entitiesData.entrySet())
			world.updateEntity(entry.getKey(), entry.getValue());
		currentTick++;
	}
	
	public void seek(int tick) throws IOException, InterruptedException, ExecutionException {
		if(tick >= duration) tick = duration - 1;
		currentTick = tick - (tick % snapshotLenght);
		Snapshot snapshot = snapshot(tick);
		snapshot.read();
		snapshot.applyDataToWorld(this);
		currentChangeSet = null;
		ChangeSet changeSet = changeSet(currentTick);
		nextChangeSet = ioExecutor.submit(() -> {
			changeSet.read();
			return changeSet;
		});
		while(currentTick < tick)
			endReplayTick();
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
