package edu.usc.thevillagers.serversideagent.env;

import edu.usc.thevillagers.serversideagent.agent.Agent;
import edu.usc.thevillagers.serversideagent.agent.AgentState;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.WorldServer;

public class EnvironmentParkour extends Environment {
	
	private static final float WIDTH = 7, LENGTH = 20;
	
	private BlockPos ref;

	public EnvironmentParkour() {
		super("Parkour", 2, 2);
	}
	
	@Override
	public void init(WorldServer world, String cmd) {
		super.init(world, cmd);
		ref = getSpawnPoint();
		System.out.println("REF: "+ref);
	}

	@Override
	protected void encodeState(Agent a, float[] stateVector) {
		stateVector[0] = (float) (a.posX - ref.getX()) / WIDTH;
		stateVector[1] = (float) (a.posZ - ref.getZ()) / LENGTH;
		stateVector[0] = 1;
		stateVector[1] = 1;
	}

	@Override
	protected void decodeAction(AgentState s, float[] actionVector) {
		s.forward = actionVector[0];
		s.strafe = actionVector[1];
	}

	@Override
	protected void step() throws Exception {
		float dz = (float) (agent.posZ - ref.getZ()) / LENGTH;
		if(agent.posY < ref.getY() - .5F) {
			reward = -100;
			done = true;
		} else {
			
			reward = dz;
		}
	}
}
