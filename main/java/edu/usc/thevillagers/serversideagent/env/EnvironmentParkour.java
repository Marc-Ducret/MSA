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

	public final int sightDist; //TODO: factorize?
	public final int sightWidth;

	public final int width, length;

	protected BlockPos ref;
	
	public EnvironmentParkour() {
		this(3, 4, 15);
	}

	public EnvironmentParkour(int sightDist, int width, int length) {
		super("Parkour", 2 + (2*sightDist+1)*(2*sightDist+1), 2);
		this.sightDist = sightDist;
		this.sightWidth = sightDist * 2 + 1;
		this.width = width;
		this.length = length;
	}

	@Override
	public void init(WorldServer world, String cmd) {
		super.init(world, cmd);
		ref = getSpawnPoint();
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
		stateVector[0] = (float) (agent.posX - ref.getX()) / width;
		stateVector[1] = (float) (agent.posZ - ref.getZ()) / length;
		BlockPos pos = agent.getPosition().add(-sightDist, -1, -sightDist);
		for(int z = 0; z < sightWidth; z++) {
			for(int x = 0; x < sightWidth; x++) {
				stateVector[2 + x + z * sightWidth] = encode(world.getBlockState(pos.add(x, 0, z)));
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
		float dz = (float) (agent.posZ - ref.getZ()) / length;
		IBlockState b = world.getBlockState(agent.getPosition().down());
		if(agent.posY < ref.getY() - .01F || time >= 100) {
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
