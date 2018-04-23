package edu.usc.thevillagers.serversideagent.recording.event;

import edu.usc.thevillagers.serversideagent.recording.WorldRecordReplayer;
import net.minecraft.block.Block;
import net.minecraft.block.state.IBlockState;
import net.minecraft.nbt.NBTTagCompound;
import net.minecraft.util.math.BlockPos;

public class RecordEventBlockChange extends RecordEvent {
	
	private BlockPos pos;
	private IBlockState state;
	
	public RecordEventBlockChange() {
	}
	
	public RecordEventBlockChange(BlockPos pos, IBlockState state) {
		this.pos = pos;
		this.state = state;
	}

	@Override
	public void replay(WorldRecordReplayer wr) {
		wr.world.setBlockState(pos, state);
	}

	@Override
	public void write(NBTTagCompound compound) {
		compound.setInteger("X", pos.getX());
		compound.setInteger("Y", pos.getY());
		compound.setInteger("Z", pos.getZ());
		compound.setInteger("State", Block.getStateId(state));
	}

	@Override
	public void read(NBTTagCompound compound) {
		pos = new BlockPos(compound.getInteger("X"), compound.getInteger("Y"), compound.getInteger("Z"));
		state = Block.getStateById(compound.getInteger("State"));
	}
}
