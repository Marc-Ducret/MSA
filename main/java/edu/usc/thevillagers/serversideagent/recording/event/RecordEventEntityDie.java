package edu.usc.thevillagers.serversideagent.recording.event;

import edu.usc.thevillagers.serversideagent.recording.WorldRecord;
import net.minecraft.nbt.NBTTagCompound;

public class RecordEventEntityDie extends RecordEvent {
	
	private int id;
	
	public RecordEventEntityDie() {
	}
	
	public RecordEventEntityDie(int id) {
		this.id = id;
	}

	@Override
	public void replay(WorldRecord wr) {
		wr.getReplayWorld().killEntity(id);
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
}
