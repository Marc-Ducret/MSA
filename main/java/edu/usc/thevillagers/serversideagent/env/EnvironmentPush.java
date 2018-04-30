package edu.usc.thevillagers.serversideagent.env;

import edu.usc.thevillagers.serversideagent.agent.Actor;
import edu.usc.thevillagers.serversideagent.agent.Agent;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.WorldServer;

public class EnvironmentPush extends Environment {

	protected BlockPos ref;
	
	private Actor actorA;
	private Actor actorB;
	
	public EnvironmentPush() {
		super(4, 2);
	}
	
	@Override
	public void init(WorldServer world) {
		super.init(world);
		ref = getOrigin();
	}
	
	@Override
	public void newActor(Actor a) {
		super.newActor(a);
		if(actorA == null) actorA = a;
		else if(actorB == null) {
			actorB = a;
			actorB.envData = actorA;
			actorA.envData = actorB;
		}
	}

	@Override
	public void encodeObservation(Agent agent, float[] stateVector) {
		stateVector[0] = (float) (agent.entity.posX - ref.getX());
		stateVector[1] = (float) (agent.entity.posZ - ref.getZ());
		Actor opponent = opponent(agent);
		stateVector[2] = (float) (opponent.entity.posX - ref.getX());
		stateVector[3] = (float) (opponent.entity.posZ - ref.getZ());
	}

	@Override
	public void decodeAction(Agent agent, float[] actionVector) {
		agent.actionState.forward = actionVector[0];
		agent.actionState.strafe = actionVector[1];
	}

	@Override
	protected void stepActor(Actor actor) throws Exception {
		if(actor.entity.posY < ref.getY() - .01F) {
			done = true;
			actor.reward = -10;
			opponent(actor).reward = 10;
		}
	}
	
	private Actor opponent(Actor a) {
		return (Actor) a.envData;
	}
}
