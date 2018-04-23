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
	
	private Future<ChangeSet> nextChangeSet;
	
	public WorldRecordReplayer(File saveFolder) {
		this.world = null;
		this.saveFolder = saveFolder;
	}
	
	@Override
	public void readInfo() throws IOException {
		super.readInfo();
		world = new ReplayWorldAccess(from, to);
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
}
