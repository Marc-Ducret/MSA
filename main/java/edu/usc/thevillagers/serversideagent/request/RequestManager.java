package edu.usc.thevillagers.serversideagent.request;

import java.io.IOException;
import java.net.ServerSocket;
import java.util.ArrayList;
import java.util.List;

import edu.usc.thevillagers.serversideagent.agent.Agent;
import edu.usc.thevillagers.serversideagent.agent.EntityAgent;
import edu.usc.thevillagers.serversideagent.env.Environment;
import edu.usc.thevillagers.serversideagent.env.EnvironmentManager;
import edu.usc.thevillagers.serversideagent.env.controller.ControllerPython;
import net.minecraft.world.WorldServer;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.fml.common.FMLCommonHandler;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;
import net.minecraftforge.fml.common.gameevent.TickEvent.Phase;
import net.minecraftforge.fml.common.gameevent.TickEvent.ServerTickEvent;

/**
 * A server that handles agent request from other processes. A request can be about a specific
 * environment instance or require to allocate a new one.
 */
public class RequestManager {
	
	private EnvironmentManager envManager;
	
	private List<Request> requests = new ArrayList<>();
	private long lastRequestTime = 0;
	
	private ServerSocket serv;
	
	public RequestManager(EnvironmentManager envManager) {
		this.envManager = envManager;
		MinecraftForge.EVENT_BUS.register(this);
	}
	
	public void startRequestServer(int port) throws IOException {
		serv = new ServerSocket(port);
		new Thread(() -> {
			while(!serv.isClosed()) {
				try {
					newSocket(new DataSocket(serv.accept()));
				} catch (Exception e) {
					System.out.println("Request failure "+e);
				}
			}
		}).start();
	}
	
	private void newSocket(DataSocket sok) throws Exception {
		sok.socket.setSoTimeout(60000);
		String envString = sok.in.readUTF();
		int parStart = envString.indexOf('[');
		String envClassName;
		float[] pars;
		if(parStart >= 0) {
			envClassName = envString.substring(0, parStart);
			String[] parsStrs = envString.substring(parStart+1, envString.indexOf(']')).split(",");
			pars = new float[parsStrs.length];
			for(int i = 0; i < parsStrs.length; i++)
				pars[i] = Float.parseFloat(parsStrs[i]);
		} else {
			envClassName = envString;
			pars = new float[] {};
		}
		Class<?> envClass = envManager.findEnvClass(envClassName);
		String envId = sok.in.readBoolean() ? null : sok.in.readUTF();
		synchronized(requests) {
			requests.add(new Request(envClass, pars, envId, sok));
			System.out.println("Received request ["+envClass.getSimpleName()+" "+envId+"]");
			lastRequestTime = System.currentTimeMillis();
		}
	}
	
	public void stopRequestServer() throws IOException {
		serv.close();
	}
	
	@SubscribeEvent
    public void serverTick(ServerTickEvent event) {
		if(event.phase == Phase.START) {
			synchronized(requests) {
				if(System.currentTimeMillis() - lastRequestTime > 400 && !requests.isEmpty()) {
					System.out.println("Processing "+requests.size()+" requests");
					while(!requests.isEmpty())
						processRequest(requests.remove(0));
				}
			}
		}
    }
	
	private void processRequest(Request req) {
		try {
			Environment env = null;
			if(req.envId != null) {
				env = envManager.getEnv(req.envId);
				if(env == null) System.out.println("Env "+req.envId+" not found, will try to allocate");
				if(env != null && env.getClass() != req.envClass)
					throw new Exception("Missmatch event types: "+env.getClass()+" | "+req.envClass);
			}
			if(env == null) {
				env = envManager.createEnv(req.envClass);
				env.readPars(req.pars);
				WorldServer world = FMLCommonHandler.instance().getMinecraftServerInstance().worlds[0];
				if(!env.tryAllocate(world)) throw new Exception("Cannot allocate "+req.envClass);
				env.init(world);
				if(req.envId != null) envManager.registerEnv(env, req.envId);
				else envManager.registerEnv(env);
			}
			
			if(req.envId == null) {
				env.setController(new ControllerPython(env, req.sok));
				req.sok.out.writeUTF(env.id);
				req.sok.out.flush();
			} else {
				String name = env.id;
				if(name.length() > 16) name = name.substring(0, 16);
				Agent a = new Agent(env, new EntityAgent(env.world, name), req.sok);
				((EntityAgent) a.entity).spawn(env.getOrigin());
				env.newActor(a);
			}
		} catch (Exception e) {
			System.out.println("Cannot process request ("+e+")");
		}
	}
	
	public static class Request {
		
		public final Class<?> envClass;
		public final float[] pars;
		public final String envId;
		public final DataSocket sok;
		
		public Request(Class<?> envClass, float[] pars, String envId, DataSocket sok) {
			this.envClass = envClass;
			this.pars = pars;
			this.envId = envId;
			this.sok = sok;
		}
	}
}
