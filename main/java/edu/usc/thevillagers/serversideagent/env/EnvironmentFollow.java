package edu.usc.thevillagers.serversideagent.env;

import edu.usc.thevillagers.serversideagent.agent.Agent;
import edu.usc.thevillagers.serversideagent.agent.AgentState;
import net.minecraft.entity.Entity;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.WorldServer;

public class EnvironmentFollow extends Environment {
	
	private Entity target;

	public EnvironmentFollow() {
		super("Follow", 2, 2);
	}
	
	@Override
	public void init(WorldServer world, String cmd) {
		super.init(world, cmd);
		BlockPos ref = getSpawnPoint();
		target = world.getClosestPlayer(ref.getX(), ref.getY(), ref.getZ(), 50, false);
	}

	@Override
	protected void encodeState(Agent a, float[] stateVector) {
		stateVector[0] = (float) (target.posX - a.posX) / 20F;
		stateVector[1] = (float) (target.posZ - a.posZ) / 20F;
	}

	@Override
	protected void decodeAction(AgentState s, float[] actionVector) {
		s.forward = actionVector[0];
		s.strafe = actionVector[1];
	}

	@Override
	protected void step() throws Exception {
		float dx = (float) (target.posX - agent.posX) / 20F;
		float dz = (float) (target.posZ - agent.posZ) / 20F;
		reward = - (dx*dx + dz*dz);
	}

	@Override
	public void reset() {
	}
}
