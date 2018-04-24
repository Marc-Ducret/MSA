package edu.usc.thevillagers.serversideagent.recording;

import java.util.ArrayList;
import java.util.List;

import edu.usc.thevillagers.serversideagent.recording.event.RecordEvent;
import net.minecraft.block.state.IBlockState;

public class SnapshotData {
	public final IBlockState[] blockStates;
	public final List<RecordEvent> spawnEvents;
	public long worldTime;
	
	public SnapshotData(int blocksBufferSize) {
		blockStates = new IBlockState[blocksBufferSize];
		spawnEvents = new ArrayList<>();
	}
}