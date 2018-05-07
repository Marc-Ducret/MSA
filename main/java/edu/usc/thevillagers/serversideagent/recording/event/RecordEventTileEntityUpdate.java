package edu.usc.thevillagers.serversideagent.recording.event;

import edu.usc.thevillagers.serversideagent.recording.WorldRecordRecorder;
import edu.usc.thevillagers.serversideagent.recording.WorldRecordReplayer;
import net.minecraft.nbt.NBTTagCompound;
import net.minecraft.util.math.BlockPos;

/**
 * Record of a TileEntity's data changing.
 * This data is computed as if the TileEntity was saved to the disk.
 */
public class RecordEventTileEntityUpdate extends RecordEvent {
	
	private BlockPos pos;
	private NBTTagCompound data;
	
	public RecordEventTileEntityUpdate() {
	}
	
	public RecordEventTileEntityUpdate(BlockPos pos, NBTTagCompound data) {
		this.pos = pos;
		this.data = data;
	}

	@Override
	public void replay(WorldRecordReplayer wr) {
		if(wr.tileEntitiesData.containsKey(pos)) WorldRecordRecorder.updateCompound(wr.tileEntitiesData.get(pos), data);
		else System.out.println("Missing tile entity at: "+pos);
	}

	@Override
	public void write(NBTTagCompound compound) {
		compound.setInteger("X", pos.getX());
		compound.setInteger("Y", pos.getY());
		compound.setInteger("Z", pos.getZ());
		compound.setTag("Data", data);
	}

	@Override
	public void read(NBTTagCompound compound) {
		pos = new BlockPos(	compound.getInteger("X"), 
							compound.getInteger("Y"), 
							compound.getInteger("Z"));
		data = compound.getCompoundTag("Data");
	}
}
