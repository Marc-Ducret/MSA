package edu.usc.thevillagers.serversideagent.env;

import edu.usc.thevillagers.serversideagent.agent.Agent;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.WorldServer;

public class EnvironmentPush extends Environment {

	protected BlockPos ref;
	
	private Agent agentA;
	private Agent agentB;
	
	public EnvironmentPush() {
		super("Push", 4, 2);
	}
	
	@Override
	public void init(WorldServer world) {
		super.init(world);
		ref = getOrigin();
	}
	
	@Override
	public void newAgent(Agent a) {
		super.newAgent(a);
		if(agentA == null) agentA = a;
		else if(agentB == null) {
			agentB = a;
			agentB.envData = agentA;
			agentA.envData = agentB;
		}
	}

	@Override
	public void encodeObservation(Agent agent, float[] stateVector) {
		stateVector[0] = (float) (agent.entity.posX - ref.getX());
		stateVector[1] = (float) (agent.entity.posZ - ref.getZ());
		Agent opponent = opponent(agent);
		stateVector[2] = (float) (opponent.entity.posX - ref.getX());
		stateVector[3] = (float) (opponent.entity.posZ - ref.getZ());
	}

	@Override
	public void decodeAction(Agent agent, float[] actionVector) {
		agent.actionState.forward = actionVector[0];
		agent.actionState.strafe = actionVector[1];
	}

	@Override
	protected void stepAgent(Agent agent) throws Exception {
		if(agent.entity.posY < ref.getY() - .01F) {
			done = true;
			agent.reward = -10;
			opponent(agent).reward = 10;
		}
	}
	
	private Agent opponent(Agent a) {
		return (Agent) a.envData;
	}
}
