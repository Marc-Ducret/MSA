package edu.usc.thevillagers.serversideagent.env;

import java.util.function.BiConsumer;
import java.util.function.Function;

import edu.usc.thevillagers.serversideagent.agent.Actor;
import edu.usc.thevillagers.serversideagent.agent.Agent;
import edu.usc.thevillagers.serversideagent.env.allocation.AllocatorEmptySpace;
import net.minecraft.block.BlockColored;
import net.minecraft.block.state.IBlockState;
import net.minecraft.init.Blocks;
import net.minecraft.item.EnumDyeColor;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.WorldServer;

public class EnvironmentTerritory extends Environment {

	public final int sightDist; //TODO: factorize?
	public final int sightWidth;

	public final int size;

	protected BlockPos ref;
	
	public EnvironmentTerritory() {
		this(5, 8);
	}

	public EnvironmentTerritory(int sightDist, int size) {
		super(4 + (2*sightDist+1)*(2*sightDist+1), 5);
		this.sightDist = sightDist;
		this.sightWidth = sightDist * 2 + 1;
		this.size = size;
		allocator = new AllocatorEmptySpace(new BlockPos(-size, -1, -size), new BlockPos(size, 2, size));
	}
	
	@Override
	public void newActor(Actor a) {
		super.newActor(a);
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
		int[] buildCd = (int[]) agent.envData;
		if(buildCd[0] > 0) {
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
	protected void step() {
		super.step();
		if(time > 100) {
			finalRewards();
			done = true;
		}
	}
	
	private void finalRewards() {
		int s = size*2 + 1;
		int[] comps = new int[s*s];
		for(int i = 0; i < s*s; i ++)
			comps[i] = i;
		
		Function<Integer, Integer> x = (i) -> i % s - size + ref.getX();
		Function<Integer, Integer> z = (i) -> i / s - size + ref.getZ();
		Function<BlockPos, Integer> index = (p) -> (p.getX() - ref.getX() + size) + s * (p.getZ() - ref.getZ() + size);
		Function<Integer, BlockPos> pos = (i) -> new BlockPos(x.apply(i), ref.getY(), z.apply(i));
		Function<BlockPos, Boolean> isIn = (p) -> 
			p.getX() >= ref.getX() - size &&
			p.getX() <= ref.getX() + size &&
			p.getZ() >= ref.getZ() - size &&
			p.getZ() <= ref.getZ() + size;
		
		Function<Integer, Integer> find = (i) -> {
			int c = i;
			while(comps[c] != c)
				c = comps[c];
			comps[i] = c;
			return c;
		};
		
		BiConsumer<Integer, Integer> join = (i, j) -> {
			int cI = find.apply(i);
			int cJ = find.apply(j);
			int c = Math.min(cI, cJ);
			comps[cI] = c;
			comps[cJ] = c;
		};
		
		for(int i = 0; i < s*s; i++) {
			BlockPos p = pos.apply(i);
			if(world.getBlockState(p).getBlock() != Blocks.AIR) continue;
			BlockPos[] neighbours = new BlockPos[] {p.east(), p.west(), p.north(), p.south()};
			for(BlockPos q : neighbours)
				if(isIn.apply(q) && world.getBlockState(q).getBlock() == Blocks.AIR)
					join.accept(i, index.apply(q));
		}
		
		int[] agentCount = new int[s*s];
		int[] blockCount = new int[s*s];
		for(int i = 0; i < s*s; i++) {
			if(world.getBlockState(pos.apply(i)).getBlock() == Blocks.AIR)
				blockCount[find.apply(i)]++;
		}
		applyToActiveActors((a) -> {
			agentCount[find.apply(index.apply(a.entity.getPosition()))]++;
		});
		applyToActiveActors((a) -> {
			int comp = find.apply(index.apply(a.entity.getPosition()));
			a.reward = blockCount[comp] / (float) agentCount[comp];
		});
	}

	@Override
	protected void stepActor(Actor actor) throws Exception {
		actor.reward = 0;
		int[] buildCd = (int[]) actor.envData;
		if(buildCd[0] > 0)
			buildCd[0]--;
		else if(actor.actionVector[4] > .9F) {
			buildCd[0] = 0;
			BlockPos p = actor.entity.getPosition().offset(actor.entity.getHorizontalFacing());
			if(world.getBlockState(p).getBlock() == Blocks.AIR) {
				world.setBlockState(p, Blocks.STAINED_GLASS.getDefaultState().withProperty(BlockColored.COLOR, EnumDyeColor.RED));
				world.setBlockState(p.up(), Blocks.STAINED_GLASS.getDefaultState().withProperty(BlockColored.COLOR, EnumDyeColor.RED));
				actor.reward = -.1F;
			} else {
				actor.reward = -.1F;
			}
		}
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
				for(int y = 0; y < 2; y ++)
					world.setBlockState(ref.add(x, y, z), 
							Math.max(Math.abs(x), Math.abs(z)) == size ? 
										Blocks.STAINED_GLASS.getDefaultState().withProperty(BlockColored.COLOR, EnumDyeColor.BLACK) :
										Blocks.AIR.getDefaultState());
			}
		}
	}
}
