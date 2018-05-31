package edu.usc.thevillagers.serversideagent.recording.event;

import edu.usc.thevillagers.serversideagent.HighLevelAction;
import edu.usc.thevillagers.serversideagent.recording.ActionListener;
import edu.usc.thevillagers.serversideagent.recording.WorldRecordReplayer;
import edu.usc.thevillagers.serversideagent.recording.WorldRecordWorker;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.nbt.NBTTagCompound;
import net.minecraft.util.math.AxisAlignedBB;
import net.minecraft.util.math.Vec3d;

/**
 * Record of an {@link HighLevelAction}.
 */
public class RecordEventAction extends RecordEvent {
	
	private HighLevelAction action;
	
	public RecordEventAction() {
	}
	
	public RecordEventAction(HighLevelAction action) {
		this.action = action;
	}

	@Override
	public void replay(WorldRecordReplayer wr) {
		for(ActionListener listener : wr.actionListeners) listener.onAction(action);
		EntityPlayer actor = (EntityPlayer) wr.getEntity(action.actorId);
		if(action.actionPhase != HighLevelAction.Phase.STOP)
			actor.swingArm(action.hand);
	}

	@Override
	public void write(NBTTagCompound compound) {
		compound.setTag("Action", action.toNBT());
	}

	@Override
	public void read(NBTTagCompound compound) {
		action = HighLevelAction.fromNBT(compound.getCompoundTag("Action"));
	}

	@Override
	public boolean isWithinBounds(WorldRecordWorker record, AxisAlignedBB bounds) {
		return record.entitiesData.containsKey(action.actorId) && 
				(action.targetEntityId < 0 || record.entitiesData.containsKey(action.targetEntityId)) &&
				(action.targetBlockPos == null || bounds.contains(new Vec3d(action.targetBlockPos).addVector(.5, .5, .5)));
	}
}
