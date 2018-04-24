package edu.usc.thevillagers.serversideagent.recording.event;

import edu.usc.thevillagers.serversideagent.HighLevelAction;
import edu.usc.thevillagers.serversideagent.recording.WorldRecordReplayer;
import net.minecraft.nbt.NBTTagCompound;

public class RecordEventAction extends RecordEvent {
	
	private HighLevelAction action;
	
	public RecordEventAction() {
	}
	
	public RecordEventAction(HighLevelAction action) {
		this.action = action;
	}

	@Override
	public void replay(WorldRecordReplayer wr) {
		//TODO do
	}

	@Override
	public void write(NBTTagCompound compound) {
		compound.setTag("Action", action.toNBT());
	}

	@Override
	public void read(NBTTagCompound compound) {
		action = HighLevelAction.fromNBT(compound);
	}
}
