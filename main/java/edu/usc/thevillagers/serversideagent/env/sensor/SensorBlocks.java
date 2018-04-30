package edu.usc.thevillagers.serversideagent.env.sensor;

import java.util.function.Function;

import edu.usc.thevillagers.serversideagent.agent.Agent;
import net.minecraft.block.state.IBlockState;
import net.minecraft.util.math.BlockPos;

public class SensorBlocks extends Sensor {

	private final BlockPos from, to;
	private final Function<IBlockState, Float> encoding;
	
	public SensorBlocks(BlockPos from, BlockPos to, Function<IBlockState, Float> encoding) {
		super(	(to.getX() - from.getX() + 1) * 
				(to.getY() - from.getY() + 1) * 
				(to.getZ() - from.getZ() + 1));
		this.from = from;
		this.to = to;
		this.encoding = encoding;
	}

	@Override
	public void sense(Agent agent) {
		int offset = 0;
		for(BlockPos p : BlockPos.getAllInBoxMutable(from, to)) {
			values[offset++] = encoding.apply(agent.entity.world.getBlockState(p));
		}
	}
}
