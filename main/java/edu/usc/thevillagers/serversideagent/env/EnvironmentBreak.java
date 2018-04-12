package edu.usc.thevillagers.serversideagent.env;

import edu.usc.thevillagers.serversideagent.agent.Agent;
import edu.usc.thevillagers.serversideagent.env.allocation.AllocatorEmptySpace;
import net.minecraft.block.BlockColored;
import net.minecraft.block.state.IBlockState;
import net.minecraft.init.Blocks;
import net.minecraft.item.EnumDyeColor;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.WorldServer;

public class EnvironmentBreak extends Environment {

	public final int sightDist; //TODO: factorize?
	public final int sightWidth;

	public final int size, sizeBlock;

	protected BlockPos ref;
	
	public EnvironmentBreak() {
		this(5, 10, 5);
		allocator = new AllocatorEmptySpace(new BlockPos(-size, -1, -size), new BlockPos(size, 2, size));
	}

	public EnvironmentBreak(int sightDist, int size, int sizeBlock) {
		super(4 + (2*sightDist+1)*(2*sightDist+1), 5);
		this.sightDist = sightDist;
		this.sightWidth = sightDist * 2 + 1;
		this.size = size;
		this.sizeBlock = sizeBlock;
	}
	
	@Override
	public void newAgent(Agent a) {
		super.newAgent(a);
		a.envData = new int[] {0}; //Break CD
	}
	
	@Override
	public void init(WorldServer world) {
		super.init(world);
		ref = getOrigin();
	}

	private int encode(IBlockState b) {
		if(b.getBlock() != Blocks.STAINED_GLASS) return 0;
		return 1;
	}

	@Override
	public void encodeObservation(Agent agent, float[] stateVector) {
		stateVector[0] = (float) (agent.entity.posX - ref.getX()) / size;
		stateVector[1] = (float) (agent.entity.posZ - ref.getZ()) / size;
		stateVector[2] = (float) (agent.entity.rotationPitch) / 360;
		stateVector[3] = (float) (agent.entity.rotationYaw) / 360;
		BlockPos pos = agent.entity.getPosition().add(-sightDist, 0, -sightDist);
		for(int z = 0; z < sightWidth; z++) {
			for(int x = 0; x < sightWidth; x++) {
				stateVector[4 + x + z * sightWidth] = encode(agent.entity.world.getBlockState(pos.add(x, 0, z)));
			}
		}
	}

	@Override
	public void decodeAction(Agent agent, float[] actionVector) {
		int[] breakCd = (int[]) agent.envData;
		if(breakCd[0] > 0) {
			agent.actionState.forward = 0;
			agent.actionState.strafe = 0;
			agent.actionState.momentumPitch = 0;
			agent.actionState.momentumYaw = 0;
		} else {
			agent.actionState.forward = actionVector[0];
			agent.actionState.strafe = actionVector[1];
			//agent.actionState.momentumPitch = actionVector[2];
			agent.actionState.momentumYaw = actionVector[3];
		}
	}

	@Override
	protected void stepAgent(Agent agent) throws Exception {
		agent.reward = 0;
		int[] breakCd = (int[]) agent.envData;
		if(breakCd[0] > 0)
			breakCd[0]--;
		else if(agent.actionVector[4] > .5) {
			breakCd[0] = 2;
			BlockPos p = agent.entity.getPosition().offset(agent.entity.getHorizontalFacing());
			if(world.getBlockState(p).getBlock() != Blocks.AIR) {
				agent.reward = 1;
				world.setBlockState(p, Blocks.AIR.getDefaultState());
				world.setBlockState(p.up(), Blocks.AIR.getDefaultState());
			} else {
				agent.reward = -.1F;
			}
		}
		if(time > 200)
			done = true;
	}
	
	@Override
	public void reset() {
		super.reset();
		generate();
	}
	
	private void generate() {
		for(int z =- size; z <= size; z++) {
			for(int x =- size; x <= size; x++) {
				world.setBlockState(ref.add(x, -1, z), Blocks.STAINED_GLASS.getDefaultState());
				if(Math.max(Math.abs(x), Math.abs(z)) <= sizeBlock && (x != 0 || z != 0))
					for(int y = 0; y < 2; y ++) {
						world.setBlockState(ref.add(x, y, z), Blocks.STAINED_GLASS.getDefaultState().withProperty(BlockColored.COLOR, EnumDyeColor.BLUE));
					}
				if(Math.max(Math.abs(x), Math.abs(z)) == size)
						world.setBlockState(ref.add(x, 1, z), Blocks.STAINED_GLASS.getDefaultState().withProperty(BlockColored.COLOR, EnumDyeColor.BLACK));
			}
		}
	}
}
