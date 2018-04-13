package edu.usc.thevillagers.serversideagent.env;

import edu.usc.thevillagers.serversideagent.agent.Agent;
import edu.usc.thevillagers.serversideagent.env.allocation.AllocatorEmptySpace;
import net.minecraft.block.BlockColored;
import net.minecraft.init.Blocks;
import net.minecraft.item.EnumDyeColor;
import net.minecraft.util.math.BlockPos;

public class EnvironmentIdle extends Environment {
	
	public EnvironmentIdle() {
		super(2, 1);
		allocator = new AllocatorEmptySpace(new BlockPos(-1, -1, -1), new BlockPos(1, 2, 1));
	}
	
	@Override
	public void reset() {
		super.reset();
		generate();
	}

	private void generate() {
		world.setBlockState(getOrigin().down(), Blocks.STAINED_GLASS.getDefaultState().withProperty(BlockColored.COLOR, EnumDyeColor.CYAN));
	}

	@Override
	public void encodeObservation(Agent agent, float[] stateVector) {
		stateVector[0] = 1;
		for(int i = 1; i < stateVector.length; i++) {
			stateVector[i] = (float) world.rand.nextGaussian();
		}
	}

	@Override
	public void decodeAction(Agent agent, float[] actionVector) {
	}

	@Override
	protected void stepAgent(Agent a) throws Exception {
		a.reward = 0;
		for(float f : a.actionVector) {
			a.reward -= f*f;
		}
		if(time >= 10)
			done = true;
	}
}
