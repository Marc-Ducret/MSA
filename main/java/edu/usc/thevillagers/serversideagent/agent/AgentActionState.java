package edu.usc.thevillagers.serversideagent.agent;

import edu.usc.thevillagers.serversideagent.HighLevelAction;
import net.minecraft.util.math.MathHelper;

public class AgentActionState {

	public float forward, strafe, momentumYaw, momentumPitch;
	public boolean jump, crouch, attack, use;
	public HighLevelAction action;
	
	public AgentActionState() {
		forward = strafe = momentumYaw = momentumPitch = 0;
		jump = crouch = attack = use = false;
		action = null;
	}
	
	public void clamp() {
		forward = MathHelper.clamp(forward, -1, +1);
		strafe = MathHelper.clamp(strafe, -1, +1);
		momentumYaw = MathHelper.clamp(momentumYaw, -1, +1);
		momentumPitch = MathHelper.clamp(momentumPitch, -1, +1);
	}
}
