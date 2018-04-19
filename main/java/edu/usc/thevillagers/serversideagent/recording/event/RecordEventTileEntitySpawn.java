package edu.usc.thevillagers.serversideagent.recording.event;

import edu.usc.thevillagers.serversideagent.recording.WorldRecord;
import net.minecraft.nbt.NBTTagCompound;
import net.minecraft.tileentity.TileEntity;
import net.minecraft.util.math.BlockPos;

public class RecordEventTileEntitySpawn extends RecordEvent {
	
	private BlockPos pos;
	private NBTTagCompound data;
	
	public RecordEventTileEntitySpawn() {
	}
	
	public RecordEventTileEntitySpawn(BlockPos pos, NBTTagCompound data) {
		this.pos = pos;
		this.data = data;
	}

	@Override
	public void replay(WorldRecord wr) {
		TileEntity tileEntity = TileEntity.create(wr.getReplayWorld().fakeWorld, data);
		tileEntity.setPos(pos);
		wr.getReplayWorld().spawnTileEntity(tileEntity);
		wr.tileEntitiesData.put(pos, data);
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
