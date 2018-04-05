package edu.usc.thevillagers.serversideagent.env;

import edu.usc.thevillagers.serversideagent.agent.Agent;
import edu.usc.thevillagers.serversideagent.agent.AgentActionState;
import net.minecraft.block.BlockColored;
import net.minecraft.block.state.IBlockState;
import net.minecraft.init.Blocks;
import net.minecraft.item.EnumDyeColor;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.WorldServer;

public class EnvironmentParkour extends Environment {
	
	private static final int SIGHT_DIST = 1; //TODO: factorize?
	private static final int SIGHT_WIDTH = SIGHT_DIST * 2 + 1;
	
	private static final float WIDTH = 3, LENGTH = 20;
	
	private BlockPos ref;

	public EnvironmentParkour() {
		super("Parkour", 2 + SIGHT_WIDTH * SIGHT_WIDTH, 2);
	}
	
	@Override
	public void init(WorldServer world, String cmd) {
		super.init(world, cmd);
		ref = getSpawnPoint();
		System.out.println("REF: "+ref);
	}

	private int encode(IBlockState b) {
		if(b.getBlock() != Blocks.CONCRETE) return -1;
		EnumDyeColor color = b.getValue(BlockColored.COLOR);
		switch(color) {
		case LIME:
			return 1;
		default:
			return 0;
		}
	}
	
	@Override
	protected void encodeState(Agent agent, float[] stateVector) {
		stateVector[0] = (float) (agent.posX - ref.getX()) / WIDTH;
		stateVector[1] = (float) (agent.posZ - ref.getZ()) / LENGTH;
		BlockPos pos = agent.getPosition().add(-SIGHT_DIST, -1, -SIGHT_DIST);
		for(int z = 0; z < SIGHT_WIDTH; z++) {
			for(int x = 0; x < SIGHT_WIDTH; x++) {
				stateVector[2 + x + z * SIGHT_WIDTH] = encode(world.getBlockState(pos.add(x, 0, z)));
			}
		}
	}

	@Override
	protected void decodeAction(AgentActionState actionState, float[] actionVector) {
		actionState.forward = actionVector[0];
		actionState.strafe = actionVector[1];
	}

	@Override
	protected void step() throws Exception {
		float dz = (float) (agent.posZ - ref.getZ()) / LENGTH;
		IBlockState b = world.getBlockState(agent.getPosition().down());
		if(agent.posY < ref.getY() - .5F || time >= 100) {
			reward = 10 * dz;
			done = true;
		} else if(b.getBlock() == Blocks.CONCRETE && b.getValue(BlockColored.COLOR) == EnumDyeColor.LIME) {
			reward = 100;
			done = true;
		} else {
			reward = - .001F;
		}
	}
}
