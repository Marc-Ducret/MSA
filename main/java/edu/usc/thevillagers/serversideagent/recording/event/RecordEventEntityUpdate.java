package edu.usc.thevillagers.serversideagent.recording.event;

import edu.usc.thevillagers.serversideagent.recording.WorldRecord;
import net.minecraft.nbt.NBTTagCompound;

public class RecordEventEntityUpdate extends RecordEvent {
	
	private int id;
	private NBTTagCompound data;
	
	public RecordEventEntityUpdate() {
	}
	
	public RecordEventEntityUpdate(int id, NBTTagCompound data) {
		this.id = id;
		this.data = data;
	}

	@Override
	public void replay(WorldRecord wr) {
		if(wr.entitiesData.containsKey(id)) WorldRecord.updateCompound(wr.entitiesData.get(id), data);
		else System.out.println("Missing entity with id: "+id);
	}

	@Override
	public void write(NBTTagCompound compound) {
		compound.setInteger("Id", id);
		compound.setTag("Data", data);
	}

	@Override
	public void read(NBTTagCompound compound) {
		id = compound.getInteger("Id");
		data = compound.getCompoundTag("Data");
	}
}
