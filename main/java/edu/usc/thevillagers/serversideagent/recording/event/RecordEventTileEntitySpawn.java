package edu.usc.thevillagers.serversideagent.recording.event;

import edu.usc.thevillagers.serversideagent.recording.WorldRecordReplayer;
import edu.usc.thevillagers.serversideagent.recording.WorldRecordWorker;
import net.minecraft.nbt.NBTTagCompound;
import net.minecraft.tileentity.TileEntity;
import net.minecraft.util.math.AxisAlignedBB;
import net.minecraft.util.math.BlockPos;

/**
 * Record of a new TileEntity.
 */
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
	public void replay(WorldRecordReplayer wr) {
		TileEntity tileEntity = TileEntity.create(wr.world, data);
		tileEntity.setWorld(wr.world);
		tileEntity.setPos(pos);
		wr.spawnTileEntity(tileEntity);
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

	@Override
	public boolean isWithinBounds(WorldRecordWorker record, AxisAlignedBB bounds) {
		return true;
	}
}
