package edu.usc.thevillagers.serversideagent.recording.event;

import edu.usc.thevillagers.serversideagent.recording.WorldRecordReplayer;
import edu.usc.thevillagers.serversideagent.recording.WorldRecordWorker;
import net.minecraft.nbt.NBTTagCompound;
import net.minecraft.util.math.AxisAlignedBB;
import net.minecraft.util.math.BlockPos;

/**
 * Record of a TileEntity's removal.
 */
public class RecordEventTileEntityDie extends RecordEvent {
	
	private BlockPos pos;
	
	public RecordEventTileEntityDie() {
	}
	
	public RecordEventTileEntityDie(BlockPos pos) {
		this.pos = pos;
	}

	@Override
	public void replay(WorldRecordReplayer wr) {
		wr.killTileEntity(pos);
		wr.tileEntitiesData.remove(pos);
	}

	@Override
	public void write(NBTTagCompound compound) {
		compound.setInteger("X", pos.getX());
		compound.setInteger("Y", pos.getY());
		compound.setInteger("Z", pos.getZ());
	}

	@Override
	public void read(NBTTagCompound compound) {
		pos = new BlockPos(	compound.getInteger("X"), 
							compound.getInteger("Y"), 
							compound.getInteger("Z"));
	}

	@Override
	public boolean isWithinBounds(WorldRecordWorker record, AxisAlignedBB bounds) {
		return true;
	}
}
