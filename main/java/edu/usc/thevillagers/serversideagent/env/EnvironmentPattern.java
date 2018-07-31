package edu.usc.thevillagers.serversideagent.env;

import edu.usc.thevillagers.serversideagent.ServerSideAgentMod;
import edu.usc.thevillagers.serversideagent.agent.Actor;
import edu.usc.thevillagers.serversideagent.env.actuator.ActuatorForwardStrafe;
import edu.usc.thevillagers.serversideagent.env.actuator.ActuatorHit;
import edu.usc.thevillagers.serversideagent.env.actuator.ActuatorLook;
import edu.usc.thevillagers.serversideagent.env.actuator.ActuatorUse;
import edu.usc.thevillagers.serversideagent.env.allocation.AllocatorEmptySpace;
import edu.usc.thevillagers.serversideagent.env.sensor.SensorRaytrace;
import edu.usc.thevillagers.serversideagent.recording.WorldRecordReplayer;
import net.minecraft.block.Block;
import net.minecraft.block.BlockColored;
import net.minecraft.block.state.IBlockState;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.init.Blocks;
import net.minecraft.init.Items;
import net.minecraft.item.EnumDyeColor;
import net.minecraft.item.Item;
import net.minecraft.item.ItemArmor;
import net.minecraft.item.ItemStack;
import net.minecraft.util.math.AxisAlignedBB;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.RayTraceResult;
import net.minecraft.util.math.Vec3d;
import net.minecraft.util.text.Style;
import net.minecraft.util.text.TextComponentString;
import net.minecraft.util.text.TextFormatting;
import net.minecraft.world.World;
import net.minecraft.world.WorldServer;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.event.world.BlockEvent;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;

public class EnvironmentPattern extends Environment {
	
	public static final String[][] PATTERNS = {
		{
			"#####",
			"#...#",
			"#...#",
			"#...#",
			"#####",
		},
		{
			"..#..",
			"..#..",
			"#####",
			"..#..",
			"..#..",
		},
		{
			"###",
			"# #",
			"###",
		}
	};
	
	public static final int HEIGHT = 4;
	public static final EnumDyeColor GROUND = EnumDyeColor.WHITE;
	public static final EnumDyeColor WALL = EnumDyeColor.GRAY;
	public static final EnumDyeColor TREE = EnumDyeColor.LIME;
	public static final EnumDyeColor[] TEAMS = 
		{EnumDyeColor.LIGHT_BLUE, EnumDyeColor.YELLOW, EnumDyeColor.RED, EnumDyeColor.PURPLE, EnumDyeColor.ORANGE, EnumDyeColor.BROWN};
	
	public static final Block BLOCK = Blocks.STAINED_GLASS;
	
	private int size;
	private int teams;
	private int trees;
	private String[] pattern;
	
	private int winner = -1;
	
	@Override
	public void readPars(float[] pars) {
		size = getRoundPar(pars, 0, 8);
		teams = getRoundPar(pars, 1, 2);
		trees = getRoundPar(pars, 2, 4);
		pattern = PATTERNS[getRoundPar(pars, 3, 2)];
		allocator = new AllocatorEmptySpace(new BlockPos(-size-1, -1, -size-1), new BlockPos(size+1, HEIGHT, size+1));
	}
	
	public static EnumDyeColor getEntityArmorColor(Entity e) {
		for(ItemStack itemStack : e.getArmorInventoryList()) {
			Item item = itemStack.getItem();
			if(item != null && item instanceof ItemArmor) {
				ItemArmor armor = (ItemArmor) item;
				int armorColor = armor.getColor(itemStack);
				for(EnumDyeColor team : TEAMS) {
					if(armorColor == (int) ServerSideAgentMod.getPrivateField(EnumDyeColor.class, "colorValue", team)) {
						return team;
					}
				}
			}
		}
		return null;
	}
	
	@Override
	protected void buildSensors() {
		sensors.add(new SensorRaytrace(24, 12, 7, 70, 2, true) {
			
			EnumDyeColor teamColor;
			
			@Override
			protected void preView(Entity viewer) {
				super.preView(viewer);
				teamColor = EnvironmentPattern.getEntityArmorColor(viewer);
			}
			
			@Override
			protected void encode(World world, Vec3d from, Vec3d dir, RayTraceResult res, float[] result) {
				// Channels: TEAM, SKY, TREE, ENTITY, DEPTH, NORMAL X, NORMAL Y
				if(teamColor == null) return;
				if(res != null) {
					switch(res.typeOfHit) {						
					case BLOCK:
						IBlockState state = world.getBlockState(res.getBlockPos());
						EnumDyeColor blockColor = state.getValue(BlockColored.COLOR);
						if(blockColor == EnvironmentPattern.GROUND || blockColor == EnvironmentPattern.WALL || blockColor == EnvironmentPattern.TREE) 
							result[0] = 0;
						else
							result[0] = blockColor == teamColor ? +1 : -1;
						result[1] = 0;
						result[2] = blockColor == EnvironmentPattern.TREE ? +1 : 0;
						result[3] = 0;
						break;
						
					case ENTITY:
						EnumDyeColor entityColor = EnvironmentPattern.getEntityArmorColor(res.entityHit);
						result[0] = entityColor == teamColor ? +1 : -1;
						result[1] = 0;
						result[2] = 0;
						result[3] = 1;
						break;
						
					default:
						throw new RuntimeException("Incorect type of hit");
					}
					result[4] = (float) res.hitVec.distanceTo(from) / dist;
					Vec3d right = dir.rotateYaw((float) Math.PI * .5F).subtract(0, dir.y, 0).normalize();
					Vec3d up = right.crossProduct(dir); 
					result[5] = (float) new Vec3d(res.sideHit.getDirectionVec()).dotProduct(right);
					result[6] = (float) new Vec3d(res.sideHit.getDirectionVec()).dotProduct(up);
				} else {
					result[0] = 0;
					result[1] = 1;
					result[2] = 0;
					result[3] = 0;
					result[4] = 1;
					result[5] = 0;
					result[6] = 0;
				}
			}
		});
	}
	
	@Override
	protected void buildActuators() {
		actuators.add(new ActuatorForwardStrafe());
		actuators.add(new ActuatorLook());
		actuators.add(new ActuatorHit());
		actuators.add(new ActuatorUse());
	}
	
	@Override
	public BlockPos getSpawnPoint(Actor a) {
		BlockPos ref = getOrigin();
		int x = world.rand.nextInt(2 * size - 1) - size + 1;
		int z = world.rand.nextInt(2 * size - 1) - size + 1;
		return ref.add(x, 0, z);
	}
	
	@Override
	public void reset() {
		super.reset();
		generate();
		winner = -1;
		
		int[] team = {0};
		applyToActiveActors((a) -> {
			a.envData = team[0];
			EnumDyeColor teamColor = TEAMS[team[0]];
			TextFormatting textColor = ServerSideAgentMod.getPrivateField(EnumDyeColor.class, "chatColor", teamColor);
			a.entity.sendMessage(new TextComponentString("Your team is "+textColor.getFriendlyName()).setStyle(new Style().setColor(textColor)));
			a.entity.inventory.clear();
			ItemArmor[] armor = {Items.LEATHER_BOOTS, Items.LEATHER_LEGGINGS, Items.LEATHER_CHESTPLATE, Items.LEATHER_HELMET};
			for(int i = 0; i < armor.length; i ++) {
				ItemStack item = new ItemStack(armor[i]);
				armor[i].setColor(item, ServerSideAgentMod.getPrivateField(EnumDyeColor.class, "colorValue", teamColor));
				a.entity.inventory.armorInventory.set(i, item);
			}
			a.entity.rotationPitch = -80 + world.rand.nextFloat() * 160;
			a.entity.rotationYaw = -180 + world.rand.nextFloat() * 360;
			a.entity.connection.setPlayerLocation(a.entity.posX, a.entity.posY, a.entity.posZ, a.entity.rotationYaw, a.entity.rotationPitch);
			if(trees == 0)
				a.entity.inventory.addItemStackToInventory(new ItemStack(BLOCK, 16));
			team[0] = (team[0]+1) % teams;
		});
	}
	
	@Override
	public void onLoad(WorldRecordReplayer record, int time) {
		super.onLoad(record, time);
		applyToActiveActors((a) -> {
			EnumDyeColor armorColor = getEntityArmorColor(a.entity);
			int teamId = -1;
			for(int i = 0; i < teams; i++)
				if(TEAMS[i] == armorColor)
					teamId = i;
			if(teamId < 0) throw new RuntimeException("Unknown team color: "+armorColor);
			a.envData = teamId;
		});
	}
	
	private void generateTree() {
		BlockPos ref = getOrigin();
		BlockPos treePos = null;
		for(int i = 0; i < 100; i++) {
			int x = world.rand.nextInt(2 * size - 1) - size + 1;
			int z = world.rand.nextInt(2 * size - 1) - size + 1;
			BlockPos pos = ref.add(x, 0, z);
			if(world.getBlockState(pos).getBlock() == Blocks.AIR && 
					world.getEntitiesWithinAABB(EntityPlayer.class, new AxisAlignedBB(pos)).isEmpty()) {
				treePos = pos;
				break;
			}
		}
		if(treePos == null) return;
		for(int i = 0; i < HEIGHT; i++) {
			world.setBlockState(treePos, BLOCK.getDefaultState().withProperty(BlockColored.COLOR, TREE));
			treePos = treePos.up();
		}
	}
	
	private void generate() {
		BlockPos ref = getOrigin();
		for(int z =- size; z <= size; z++)
			for(int x =- size; x <= size; x++) {
				boolean wall = Math.abs(x) == size || Math.abs(z) == size;
				for(int y = 0; y < HEIGHT; y++)
						world.setBlockState(ref.add(x, y, z), 
								wall ? BLOCK.getDefaultState().withProperty(BlockColored.COLOR, WALL) :
									   Blocks.AIR.getDefaultState());
				world.setBlockState(ref.add(x, -1, z), 
						BLOCK.getDefaultState().withProperty(BlockColored.COLOR, GROUND));
			}
		
		for(int i = 0 ; i < trees; i++)
			generateTree();
	}

	@Override
	protected void stepActor(Actor a) throws Exception {
		a.entity.heal(50);
		a.reward = -.01F;
		if(winner >= 0) {
			done = true;
			TextFormatting winnerText = ServerSideAgentMod.getPrivateField(EnumDyeColor.class, "chatColor", TEAMS[winner]);
			a.entity.sendMessage(new TextComponentString(winnerText.getFriendlyName()+" won").setStyle(new Style().setColor(winnerText)));
			if(((int)a.envData) == winner) {
				a.reward = 10;
				a.entity.sendMessage(new TextComponentString("You won!"));
				a.entity.sendMessage(new TextComponentString("Time = "+time));
			} else {
				a.reward = -10;
				a.entity.sendMessage(new TextComponentString("You lost!"));
				a.entity.sendMessage(new TextComponentString("Time = "+time));
			}
		} else if(time >= 20 * 120) {
			done = true;
			a.entity.sendMessage(new TextComponentString("Time ran out"));
		}
	}
	
	@Override
	protected void step() {
		super.step();
	}
	
	@Override
	public void init(WorldServer world) {
		super.init(world);
		MinecraftForge.EVENT_BUS.register(this);
	}
	
	@Override
	public void terminate() {
		super.terminate();
		MinecraftForge.EVENT_BUS.unregister(this);
	}
	
	private Actor getActor(EntityPlayer e) {
		Actor[] actor = {null};
		applyToActiveActors((a) -> {
			if(a.entity == e) actor[0] = a;
		});
		return actor[0];
	}
	
	private boolean isPlayerActive(EntityPlayer e) {
		return getActor(e) != null;
	}
	
	private boolean checkPattern(BlockPos start, EnumDyeColor color) {
		int patternSize = pattern.length;
		for(int z = 0; z < patternSize; z++)
			for(int x = 0; x < patternSize; x++)
				if(pattern[z].charAt(x) != '.') {
					IBlockState state = world.getBlockState(start.add(x, 0, z));
					if(pattern[z].charAt(x) == ' ') {
						if(state.getBlock() != Blocks.AIR) return false;
					} else if(state.getBlock() != BLOCK || state.getValue(BlockColored.COLOR) != color)
						return false;
				}
		return true;
	}
	
	@SubscribeEvent
	public void blockPlaced(BlockEvent.PlaceEvent event) {
		Actor actor = getActor(event.getPlayer());
		if(actor == null) return;
		if(event.getPos().getY() != getOrigin().getY()) {
			event.setCanceled(true);
		} else {
			EnumDyeColor teamColor = TEAMS[(int)actor.envData];
			world.setBlockState(event.getPos(), 
					BLOCK.getDefaultState().withProperty(BlockColored.COLOR, teamColor));
			int patternSize = pattern.length;
			for(int patternZ = 0; patternZ < patternSize; patternZ++) {
				for(int patternX = 0; patternX < patternSize; patternX++) {
					if(checkPattern(event.getPos().add(-patternX, 0, -patternZ), teamColor)) {
						winner = (int)actor.envData;
						System.out.println(teamColor+" win "+" in "+name);
					}
				}
			}
		}
	}
	
	private boolean isTree(IBlockState state) {
		return state.getBlock() == BLOCK && state.getValue(BlockColored.COLOR) == TREE;
	}

	@SubscribeEvent
	public void blockBroken(BlockEvent.BreakEvent event) {
		if(!isPlayerActive(event.getPlayer())) return;
		if(!isTree(event.getState()))
			event.setCanceled(true);
		else {
			BlockPos p = event.getPos();
			while(isTree(world.getBlockState(p.down()))) p = p.down();
			while(isTree(world.getBlockState(p))) {
				world.setBlockState(p, Blocks.AIR.getDefaultState());
				p = p.up();
			}
			event.getPlayer().inventory.addItemStackToInventory(new ItemStack(BLOCK, 2));
			generateTree();
		}
	}
}
