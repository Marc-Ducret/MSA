package edu.usc.thevillagers.serversideagent.env;

import java.util.ArrayList;
import java.util.List;

import edu.usc.thevillagers.serversideagent.agent.Agent;
import edu.usc.thevillagers.serversideagent.env.allocation.AllocatorEmptySpace;
import net.minecraft.block.BlockColored;
import net.minecraft.init.Blocks;
import net.minecraft.item.EnumDyeColor;
import net.minecraft.util.math.BlockPos;

public class EnvironmentLandmarks extends Environment {
	

	private final int nLandmarks;
	private final int nAgents;
	private final int size;
	
	private final List<Agent> agents;
	private final BlockPos[] landmarks;
	
	public EnvironmentLandmarks() {
		this(5, 5);
	}
	
	public EnvironmentLandmarks(int nLandmarks, int nAgents) {
		super((nLandmarks + nAgents) * 2, 2);
		this.nLandmarks = nLandmarks;
		this.nAgents = nAgents;
		size = 8;
		agents = new ArrayList<>(nAgents);
		landmarks = new BlockPos[nLandmarks];
		allocator = new AllocatorEmptySpace(new BlockPos(-size, -1, -size), new BlockPos(size, 2, size));
	}
	
	@Override
	public void reset() {
		super.reset();
		agents.clear();
		applyToActiveAgents((a) -> {
			if(agents.size() < nAgents) agents.add(a);
		});
		for(int i = 0; i < nLandmarks; i++) {
			landmarks[i] = getOrigin().add(world.rand.nextInt(2*size - 1) - size + 1, -1, world.rand.nextInt(2*size - 1) - size + 1);
		}
		generate();
	}
	
	private void generate() {
		BlockPos ref = getOrigin();
		for(int z =- size; z <= size; z++) {
			for(int x =- size; x <= size; x++) {
				world.setBlockState(ref.add(x, -1, z), Blocks.STAINED_GLASS.getDefaultState());
				for(int y = 0; y < 2; y ++)
					world.setBlockState(ref.add(x, y, z), 
							Math.max(Math.abs(x), Math.abs(z)) == size ? 
										Blocks.STAINED_GLASS.getDefaultState().withProperty(BlockColored.COLOR, EnumDyeColor.BLACK) :
										Blocks.AIR.getDefaultState());
			}
		}
		for(BlockPos p : landmarks) {
			world.setBlockState(p, Blocks.STAINED_GLASS.getDefaultState().withProperty(BlockColored.COLOR, EnumDyeColor.BLUE));
		}
	}

	@Override
	public void encodeObservation(Agent agent, float[] stateVector) {
		BlockPos p = agent.entity.getPosition();
		int offset = agents.indexOf(agent);
		for(int i = 0; i < nAgents; i ++) {
			int iAgent = (i + offset) % nAgents;
			if(agents.size() > iAgent) {
				BlockPos aP = agents.get(iAgent).entity.getPosition();
				stateVector[i * 2 + 0] = aP.getX() - p.getX();
				stateVector[i * 2 + 1] = aP.getZ() - p.getZ();
			}
		}
		for(int i = 0; i < nLandmarks; i++) {
			BlockPos lP = landmarks[i];
			stateVector[2 * nAgents + i * 2 + 0] = lP.getX() - p.getX();
			stateVector[2 * nAgents + i * 2 + 1] = lP.getZ() - p.getZ();
		}
	}

	@Override
	public void decodeAction(Agent agent, float[] actionVector) {
		agent.actionState.forward = actionVector[0];
		agent.actionState.strafe = actionVector[1];
	}

	@Override
	protected void stepAgent(Agent a) throws Exception {
		
	}
	
	@Override
	protected void step() {
		super.step();
		float reward = 0;
		for(BlockPos landmark : landmarks) {
			float distSq = size*size;
			for(Agent a : agents) distSq = Math.min(distSq, (float) landmark.distanceSq(a.entity.getPosition()));
			reward -= distSq * .1F;
		}
		for(Agent a : agents)
			a.reward = reward;
		if(time >= 199)
			done = true;
	}

}
