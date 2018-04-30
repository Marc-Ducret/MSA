package edu.usc.thevillagers.serversideagent.env;

import edu.usc.thevillagers.serversideagent.agent.Actor;
import edu.usc.thevillagers.serversideagent.agent.Agent;
import edu.usc.thevillagers.serversideagent.env.allocation.AllocatorEmptySpace;
import net.minecraft.block.BlockColored;
import net.minecraft.block.state.IBlockState;
import net.minecraft.init.Blocks;
import net.minecraft.item.EnumDyeColor;
import net.minecraft.util.EnumFacing;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.WorldServer;

public class EnvironmentParkourJump extends Environment {

	public final int sightDist; //TODO: factorize?
	public final int sightWidth;

	public final int width, length;

	protected BlockPos ref;
	
	public EnvironmentParkourJump() {
		this(5, 4, 15);
	}

	public EnvironmentParkourJump(int sightDist, int width, int length) {
		super(3 + (2*sightDist+1)*(2*sightDist+1), 3);
		this.sightDist = sightDist;
		this.sightWidth = sightDist * 2 + 1;
		this.width = width;
		this.length = length;
		allocator = new AllocatorEmptySpace(new BlockPos(-width-1, -1, -2), new BlockPos(width+1, 2, length+1));
	}
	
	@Override
	public void init(WorldServer world) {
		super.init(world);
		ref = getOrigin();
	}

	private int encode(IBlockState b) {
		if(b.getBlock() != Blocks.STAINED_GLASS) return -1;
		EnumDyeColor color = b.getValue(BlockColored.COLOR);
		switch(color) {
		case LIME:
			return 10;
		default:
			return 1;
		}
	}

	@Override
	public void encodeObservation(Agent agent, float[] stateVector) {
		stateVector[0] = (float) (agent.entity.posX - ref.getX()) / width;
		stateVector[1] = (float) (agent.entity.posZ - ref.getZ()) / length;
		stateVector[2] = (float) (agent.entity.posY - ref.getY());
		BlockPos pos = agent.entity.getPosition().add(-sightDist, -1, -sightDist);
		for(int z = 0; z < sightWidth; z++) {
			for(int x = 0; x < sightWidth; x++) {
				stateVector[3 + x + z * sightWidth] = encode(agent.entity.world.getBlockState(pos.add(x, 0, z)));
			}
		}
	}

	@Override
	public void decodeAction(Agent agent, float[] actionVector) {
		agent.actionState.forward = actionVector[0];
		agent.actionState.strafe = actionVector[1];
		agent.actionState.jump = actionVector[2] > .5;
	}

	@Override
	protected void stepActor(Actor actor) throws Exception {
		float dz = (float) (actor.entity.posZ - ref.getZ()) / length;
		IBlockState b = world.getBlockState(actor.entity.getPosition().down());
		if(actor.entity.posY < ref.getY() - .01F || time >= 100) {
			actor.reward = 10 * dz;
			done = true;
		} else if(b.getBlock() == Blocks.STAINED_GLASS && b.getValue(BlockColored.COLOR) == EnumDyeColor.LIME) {
			actor.reward = 100;
			done = true;
		} else {
			actor.reward = - .001F;
		}
	}
	
	@Override
	public void reset() {
		super.reset();
		generate();
	}
	
	private void generate() {
		for(int z = 1; z <= length; z++) {
			for(int x =- width; x <= width; x++) {
				world.setBlockState(ref.add(x, -1, z), Blocks.AIR.getDefaultState());
			}
		}
		BlockPos pos = ref.add(0, -1, 1);
		int blocks = length + world.rand.nextInt(2 * width);
		EnumFacing[] dirs = new EnumFacing[] {EnumFacing.EAST, EnumFacing.WEST, EnumFacing.SOUTH};
		for(int i = 0; i < blocks; i ++) {
			if(i % 2 == 0) world.setBlockState(pos, Blocks.STAINED_GLASS.getDefaultState());
			EnumFacing dir = dirs[world.rand.nextInt(dirs.length)];
			BlockPos p = pos.offset(dir);
			if(p.getX() >= ref.getX() - width && p.getX() <= ref.getX() + width && p.getZ() > ref.getZ() && p.getZ() <= ref.getZ() + length)
				pos = p;
		}
		world.setBlockState(pos, Blocks.STAINED_GLASS.getDefaultState().withProperty(BlockColored.COLOR, EnumDyeColor.LIME));
	}
}
