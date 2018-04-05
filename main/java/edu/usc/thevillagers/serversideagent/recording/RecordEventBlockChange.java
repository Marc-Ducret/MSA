package edu.usc.thevillagers.serversideagent.recording;

import net.minecraft.block.state.IBlockState;
import net.minecraft.util.math.BlockPos;

public class RecordEventBlockChange extends RecordEvent {
	
	public final BlockPos pos;
	public final IBlockState state;
	
	public RecordEventBlockChange(BlockPos pos, IBlockState state) {
		this.pos = pos;
		this.state = state;
	}

	@Override
	public void replay(WorldRecord wr) {
		wr.world.setBlockState(pos, state);
	}
}
