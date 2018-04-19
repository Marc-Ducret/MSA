package edu.usc.thevillagers.serversideagent.recording.event;

import edu.usc.thevillagers.serversideagent.recording.WorldRecord;
import net.minecraft.nbt.NBTTagCompound;
import net.minecraft.util.math.BlockPos;

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
	public void replay(WorldRecord wr) {
		if(wr.tileEntitiesData.containsKey(pos)) WorldRecord.updateCompound(wr.tileEntitiesData.get(pos), data);
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
