package edu.usc.thevillagers.serversideagent.recording;

import java.io.File;
import java.io.IOException;
import java.util.Map.Entry;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import edu.usc.thevillagers.serversideagent.recording.event.RecordEvent;
import net.minecraft.nbt.NBTTagCompound;
import net.minecraft.util.math.BlockPos;

public class WorldRecordReplayer extends WorldRecordWorker {
	
	public ReplayWorldAccess world;

	public long worldTimeOffset;
	
	private Future<ChangeSet> nextChangeSet;
	
	public WorldRecordReplayer(File saveFolder) {
		this.world = null;
		this.saveFolder = saveFolder;
	}
	
	@Override
	public void readInfo() throws IOException {
		super.readInfo();
		world = createWorldAccess();
	}
	
	protected ReplayWorldAccess createWorldAccess() {
		return new ReplayWorldAccess(from, to);
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
		for(Entry<Integer, NBTTagCompound> entry : entitiesData.entrySet())
			world.updateEntity(entry.getKey(), entry.getValue());
		for(Entry<BlockPos, NBTTagCompound> entry : tileEntitiesData.entrySet())
			world.updateTileEntity(entry.getKey(), entry.getValue());
		currentTick++;
		world.fakeWorld.setTotalWorldTime(worldTimeOffset + currentTick);
		world.fakeWorld.setWorldTime((worldTimeOffset + currentTick) % 24000);
	}
	
	public void seek(int tick) throws IOException, InterruptedException, ExecutionException {
		long start = System.currentTimeMillis();
		
		if(tick >= duration) tick = duration - 1;
		currentTick = tick - (tick % snapshotLenght);
		
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
}
