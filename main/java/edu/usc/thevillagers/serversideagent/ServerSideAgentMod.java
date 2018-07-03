package edu.usc.thevillagers.serversideagent;

import java.io.IOException;
import java.lang.reflect.Field;
import java.util.List;

import com.google.common.base.Predicate;
import com.google.common.base.Predicates;

import edu.usc.thevillagers.serversideagent.command.CommandCompile;
import edu.usc.thevillagers.serversideagent.command.CommandConstant;
import edu.usc.thevillagers.serversideagent.command.CommandEnvironment;
import edu.usc.thevillagers.serversideagent.command.CommandExecute;
import edu.usc.thevillagers.serversideagent.command.CommandExperiment;
import edu.usc.thevillagers.serversideagent.command.CommandFastTick;
import edu.usc.thevillagers.serversideagent.command.CommandRecord;
import edu.usc.thevillagers.serversideagent.command.CommandTPS;
import edu.usc.thevillagers.serversideagent.env.EnvironmentManager;
import edu.usc.thevillagers.serversideagent.proxy.Proxy;
import edu.usc.thevillagers.serversideagent.recording.WorldRecordReplayer;
import edu.usc.thevillagers.serversideagent.request.RequestManager;
import net.minecraft.entity.Entity;
import net.minecraft.util.EntitySelectors;
import net.minecraft.util.math.AxisAlignedBB;
import net.minecraft.util.math.RayTraceResult;
import net.minecraft.util.math.RayTraceResult.Type;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.DimensionType;
import net.minecraft.world.World;
import net.minecraftforge.common.DimensionManager;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.common.Mod.EventHandler;
import net.minecraftforge.fml.common.SidedProxy;
import net.minecraftforge.fml.common.event.FMLInitializationEvent;
import net.minecraftforge.fml.common.event.FMLPreInitializationEvent;
import net.minecraftforge.fml.common.event.FMLServerStartingEvent;
import net.minecraftforge.fml.common.event.FMLServerStoppingEvent;

/**
 * Main class of the Mod that is loaded by Forge on startup.
 */
@Mod(modid = ServerSideAgentMod.MODID, name = ServerSideAgentMod.NAME, version = ServerSideAgentMod.VERSION, acceptableRemoteVersions = "*")
public class ServerSideAgentMod {
    public static final String MODID = "serversideagent";
    public static final String NAME = "Server Side Agent";
    public static final String VERSION = "1.0";
    
    public EnvironmentManager envManager;
    private RequestManager reqManager;
    
    @SidedProxy(clientSide = "edu.usc.thevillagers.serversideagent.proxy.ClientProxy", serverSide = "edu.usc.thevillagers.serversideagent.proxy.ServerProxy")
    public static Proxy proxy;
	public static ServerSideAgentMod instance;

    @EventHandler
    public void preInit(FMLPreInitializationEvent event) {
    	instance = this;
    }

    @EventHandler
    public void init(FMLInitializationEvent event) {
    	MinecraftForge.EVENT_BUS.register(this);
    	MinecraftForge.EVENT_BUS.register(proxy);
    }
    
    @EventHandler
    public void serverLoad(FMLServerStartingEvent event) throws IOException {
    	envManager = new EnvironmentManager();
    	reqManager = new RequestManager(envManager);
    	reqManager.startRequestServer(1337);
    	
    	event.registerServerCommand(new CommandEnvironment(envManager));
    	event.registerServerCommand(new CommandFastTick());
    	event.registerServerCommand(new CommandTPS());
    	event.registerServerCommand(new CommandRecord());
    	event.registerServerCommand(new CommandCompile(envManager));
    	event.registerServerCommand(new CommandExecute());
    	event.registerServerCommand(new CommandExperiment(envManager));
    	event.registerServerCommand(new CommandConstant());
    	
    	DimensionManager.registerDimension(WorldRecordReplayer.DUMMY_DIMENSION, DimensionType.OVERWORLD);
    }
    
    @EventHandler
    public void serverClosing(FMLServerStoppingEvent event) throws IOException {
    	envManager.clearEnvs();
    	reqManager.stopRequestServer();
    }
    
    public static <T> T getPrivateField(Class<?> clazz, String fieldName, Object obj) {
		try {
			Field f = clazz.getDeclaredField(fieldName);
			f.setAccessible(true);
			return (T) f.get(obj);
		} catch (Exception e) {
			return null;
		}
    }
    
    public static <T> void setPrivateField(Class<?> clazz, String fieldName, Object obj, T value) {
		try {
			Field f = clazz.getDeclaredField(fieldName);
			f.setAccessible(true);
			f.set(obj, value);
		} catch (Exception e) {
		}
    }
    
    public static RayTraceResult rayTrace(World world, Vec3d from, Vec3d to) {
    	return rayTrace(world, from, to, false);
    }
    
	public static RayTraceResult rayTrace(World world, Vec3d from, Vec3d to, boolean hitEntitites) {
    	return rayTrace(world, from, to, hitEntitites, null);
    }
	
	public static RayTraceResult rayTrace(World world, Vec3d from, Vec3d to, boolean hitEntitites, Entity viewer) {
		Vec3d diff = to.subtract(from);
		RayTraceResult closest = world.rayTraceBlocks(from, to);
		if(hitEntitites) {
			AxisAlignedBB bounds = viewer == null ? new AxisAlignedBB(from.x, from.y, from.z, from.x, from.y, from.z) : viewer.getEntityBoundingBox();
			bounds = bounds.expand(diff.x, diff.y, diff.z).grow(1);
			List<Entity> entities = world.getEntitiesInAABBexcluding(viewer, bounds, Predicates.and(EntitySelectors.NOT_SPECTATING, new Predicate<Entity>() {
				@Override
				public boolean apply(Entity e) {
					return e != null && e.canBeCollidedWith();
				}
			}));
			for(Entity e : entities) {
				RayTraceResult hit = e.getEntityBoundingBox().grow(e.getCollisionBorderSize()).calculateIntercept(from, to);
				if(hit != null &&
						(closest == null || hit.hitVec.squareDistanceTo(from) < closest.hitVec.squareDistanceTo(from))) {
					hit.typeOfHit = Type.ENTITY;
					hit.entityHit = e;
					closest = hit;
				}
			}
		}
        return closest;
	}
}
