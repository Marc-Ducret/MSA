package edu.usc.thevillagers.serversideagent.recording.event;

import edu.usc.thevillagers.serversideagent.recording.WorldRecordReplayer;
import edu.usc.thevillagers.serversideagent.recording.WorldRecordWorker;
import net.minecraft.nbt.NBTTagCompound;
import net.minecraft.util.math.AxisAlignedBB;

/**
 * Record of an entity's death.
 */
public class RecordEventEntityDie extends RecordEvent {
	
	private int id;
	
	public RecordEventEntityDie() {
	}
	
	public RecordEventEntityDie(int id) {
		this.id = id;
	}

	@Override
	public void replay(WorldRecordReplayer wr) {
		wr.killEntity(id);
		wr.entitiesData.remove(id);
	}

	@Override
	public void write(NBTTagCompound compound) {
		compound.setInteger("Id", id);
	}

	@Override
	public void read(NBTTagCompound compound) {
		id = compound.getInteger("Id");
	}

	@Override
	public boolean isWithinBounds(WorldRecordWorker record, AxisAlignedBB bounds) {
		return true;
	}
}
