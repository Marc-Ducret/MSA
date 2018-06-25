package edu.usc.thevillagers.serversideagent.gui;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Method;
import java.util.List;
import java.util.concurrent.PriorityBlockingQueue;

import org.lwjgl.input.Keyboard;
import org.lwjgl.input.Mouse;
import org.lwjgl.opengl.GL11;

import edu.usc.thevillagers.serversideagent.ServerSideAgentMod;
import edu.usc.thevillagers.serversideagent.env.sensor.SensorRaytrace;
import edu.usc.thevillagers.serversideagent.recording.WorldRecordReplayerClient;
import net.minecraft.block.BlockColored;
import net.minecraft.block.state.IBlockState;
import net.minecraft.client.gui.GuiButton;
import net.minecraft.client.gui.GuiScreen;
import net.minecraft.client.renderer.GlStateManager;
import net.minecraft.client.renderer.RenderGlobal;
import net.minecraft.client.renderer.chunk.ChunkCompileTaskGenerator;
import net.minecraft.client.renderer.chunk.ChunkRenderDispatcher;
import net.minecraft.client.renderer.chunk.ChunkRenderWorker;
import net.minecraft.client.renderer.chunk.CompiledChunk;
import net.minecraft.client.renderer.chunk.RenderChunk;
import net.minecraft.entity.Entity;
import net.minecraft.entity.EntityLivingBase;
import net.minecraft.entity.MoverType;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.init.Blocks;
import net.minecraft.init.Items;
import net.minecraft.item.EnumDyeColor;
import net.minecraft.item.ItemStack;
import net.minecraft.util.EnumHand;
import net.minecraft.util.math.RayTraceResult;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.World;
import net.minecraftforge.fml.client.config.GuiSlider;

/**
 * Display for {@link WorldRecordReplayerClient}
 */
public class GuiReplay extends GuiScreen {
	
	private WorldRecordReplayerClient record;
	private int speed;
	
	private Entity followedEntity;
	
	private GuiSlider seekSlider;
	
	public GuiReplay(File recordFolder) {
		record = new WorldRecordReplayerClient(recordFolder);
		speed = 1;
	}

	@Override
	public void initGui() {
		super.initGui();
		try {
			if(record.world == null) {
				record.readInfo();
				
				record.seek(0);
				Vec3d pos = new Vec3d(record.from.add(record.to)).scale(.5);
				record.player.setPosition(pos.x, pos.y, pos.z);
				
				ChunkRenderDispatcher dispatcher =
						ServerSideAgentMod.<ChunkRenderDispatcher>getPrivateField(RenderGlobal.class, "renderDispatcher", mc.renderGlobal);
				Method processTask = ChunkRenderWorker.class.getDeclaredMethod("processTask", ChunkCompileTaskGenerator.class);
				processTask.setAccessible(true);
				ServerSideAgentMod.<PriorityBlockingQueue<ChunkCompileTaskGenerator>>setPrivateField(
						ChunkRenderDispatcher.class, "queueChunkUpdates", dispatcher,
						new PriorityBlockingQueue<ChunkCompileTaskGenerator>() {
							private static final long serialVersionUID = 1L;
							
							@Override
							public boolean offer(ChunkCompileTaskGenerator e) {
								try {
									processTask.invoke(ServerSideAgentMod.getPrivateField(ChunkRenderDispatcher.class, "renderWorker", dispatcher), e);
								} catch (Exception e1) {
									e1.printStackTrace();
									return false;
								}
								return true;
							}
						});
			}
		} catch(Exception e) {
			e.printStackTrace();
		}
		addButton(new GuiButton(0, 0, height-20, 20, 20, "<"));
		addButton(new GuiButton(1, 50, height-20, 20, 20, ">"));
		addButton(new GuiButton(2, 75, height-20, 20, 20, "+1"));
		seekSlider = new GuiSlider(3, width/2 - 100, height-20, 200, 20, "", " s", 0, record.duration / 20F, 10, true, true) {
			@Override
			public void mouseReleased(int x, int y) {
				super.mouseReleased(x, y);
				mc.addScheduledTask(() -> {
					try {
						followedEntity = null;
						record.seek((int) Math.round(getValue() * 20));
					} catch (Exception e) {
						e.printStackTrace();
					}
				});
			}
		};
		addButton(seekSlider);
	}
	
	@Override
	protected void actionPerformed(GuiButton button) throws IOException {
		super.actionPerformed(button);
		switch(button.id) {
		case 0:
			if(speed > 1)
				speed /= 2;
			else
				speed = 0;
			break;
		case 1:
			if(speed < 2048)
				speed *= 2;
			if(speed == 0)
				speed = 1;
			break;
		case 2:
			try {
				record.endReplayTick();
			} catch (Exception e) {
				throw new RuntimeException("Replay tick failure", e);
			}
			break;
		}
	}
	
	@Override
	protected void mouseClicked(int mouseX, int mouseY, int mouseButton) throws IOException {
		super.mouseClicked(mouseX, mouseY, mouseButton);
		for(GuiButton b : buttonList)
			if(b.isMouseOver())
				return;
		Mouse.setGrabbed(mouseButton == 0);
	}
	
	@Override
	protected void keyTyped(char typedChar, int keyCode) throws IOException {
		super.keyTyped(typedChar, keyCode);
		if(keyCode == mc.gameSettings.keyBindInventory.getKeyCode()) {
			if(followedEntity == null) {
				Vec3d rayStart = record.player.getPositionEyes(1);
				Vec3d rayEnd = rayStart.add(record.player.getLookVec().scale(5));
				for(Entity e : record.world.getLoadedEntityList()) { //TODO use world ray trace ;)
					if(e.getEntityBoundingBox() != null && e.getEntityBoundingBox().calculateIntercept(rayStart, rayEnd) != null)
						followedEntity = e;
				}
			} else {
				followedEntity = null;
			}
		}
	}
	
	@Override
	public void onGuiClosed() {
		super.onGuiClosed();
		Mouse.setGrabbed(false);
	}
	
	@Override
	public void updateScreen() {
		super.updateScreen();
		if(followedEntity != null) {
			Vec3d pos = followedEntity.getPositionVector();
			record.player.setPositionAndRotation(pos.x, pos.y, pos.z, followedEntity.rotationYaw, followedEntity.rotationPitch);
		} else {
			if(Mouse.isGrabbed()) {
				Vec3d move = Vec3d.ZERO;
				if(Keyboard.isKeyDown(mc.gameSettings.keyBindForward.getKeyCode())) move = move.addVector(+0, +0, -1);
				if(Keyboard.isKeyDown(mc.gameSettings.keyBindBack	.getKeyCode())) move = move.addVector(+0, +0, +1);
				if(Keyboard.isKeyDown(mc.gameSettings.keyBindRight	.getKeyCode())) move = move.addVector(+1, +0, +0);
				if(Keyboard.isKeyDown(mc.gameSettings.keyBindLeft	.getKeyCode())) move = move.addVector(-1, +0, +0);
				if(Keyboard.isKeyDown(mc.gameSettings.keyBindJump	.getKeyCode())) move = move.addVector(+0, +1, +0);
				if(Keyboard.isKeyDown(mc.gameSettings.keyBindSneak	.getKeyCode())) move = move.addVector(+0, -1, +0);
				move = move.normalize();
				move = move.rotateYaw((float)Math.toRadians(180-record.player.rotationYaw)).scale(.8);
				record.player.noClip = true;
				record.player.move(MoverType.SELF, move.x, move.y, move.z);
				record.player.turn(Mouse.getDX(), Mouse.getDY());
				Mouse.setCursorPosition(mc.displayWidth / 2, mc.displayHeight / 2);
			}
		}
		for(EnumHand hand : EnumHand.values()) {
			ItemStack item = ItemStack.EMPTY;
			record.player.setInvisible(true);
			if(followedEntity != null && followedEntity instanceof EntityPlayer) {
				record.player.setInvisible(false);
				item = ((EntityPlayer)followedEntity).getHeldItem(hand);
			}
			if(!ItemStack.areItemsEqual(item, record.player.getHeldItem(hand))) {
				record.player.setHeldItem(hand, item);
			}
		}
		
		try {
			Method method = EntityLivingBase.class.getDeclaredMethod("updateEntityActionState");
			method.setAccessible(true);
			for(int i = 0; i < speed; i ++) {
				record.endReplayTick();
				for(Entity e : record.world.getLoadedEntityList()) {
					if(e instanceof EntityPlayer) {
						EntityPlayer p = (EntityPlayer) e;
						method.invoke(p);
					}
				}
				if(followedEntity != null) {
					mc.setRenderViewEntity(record.player);
					record.player.updateEntityActionState();
					if(followedEntity instanceof EntityLivingBase) {
						EntityLivingBase e = (EntityLivingBase)followedEntity;
						record.player.swingProgress = e.swingProgress;
						record.player.swingingHand = e.swingingHand;
						record.player.swingProgressInt = e.swingProgressInt;
						record.player.isSwingInProgress = e.isSwingInProgress;
					}
				}
			}
			record.world.updateEntityWithOptionalForce(record.player, false);
			seekSlider.setValue(record.currentTick / 20F);
			seekSlider.updateSlider();
		} catch (Exception e) {
			throw new RuntimeException("Replay tick failure", e);
		}
		
		mc.getTextureMapBlocks().tick();
	}
	
	@Override
	public void drawScreen(int mouseX, int mouseY, float partialTicks) {
		drawDefaultBackground();
		
		mc.world = record.world;
		mc.player = record.player;
		mc.playerController = record.playerController;
		mc.setRenderViewEntity(followedEntity != null ? followedEntity : record.player);
		List<Object> renderInfos =
				ServerSideAgentMod.getPrivateField(RenderGlobal.class, "renderInfos", mc.renderGlobal);
		for(Object obj : renderInfos) {
			RenderChunk renderChunk = ServerSideAgentMod.getPrivateField(obj.getClass(), "renderChunk", obj);
			if(renderChunk.compiledChunk == CompiledChunk.DUMMY)
				renderChunk.setNeedsUpdate(true);
		}
		
		try {
			GlStateManager.matrixMode(GL11.GL_PROJECTION);
			GlStateManager.pushMatrix();
			GlStateManager.loadIdentity();
			GlStateManager.matrixMode(GL11.GL_MODELVIEW);
			GlStateManager.pushMatrix();
			GlStateManager.loadIdentity();
			mc.entityRenderer.updateRenderer();
			mc.entityRenderer.renderWorld(1, System.nanoTime() + 0); //TODO interpolation?
	        GlStateManager.matrixMode(GL11.GL_PROJECTION);
			GlStateManager.popMatrix();
			GlStateManager.matrixMode(GL11.GL_MODELVIEW);
			GlStateManager.popMatrix();
			GlStateManager.disableDepth();
			
			if(Keyboard.isKeyDown(Keyboard.KEY_V)) drawDebugVision();
		} catch(Exception e) {
			e.printStackTrace();
		}
		super.drawScreen(mouseX, mouseY, partialTicks);
		drawCenteredString(mc.fontRenderer, speed+"*", 35, height-14, 0xFFFFFF);
		if(followedEntity != null) {
			drawCenteredString(mc.fontRenderer, followedEntity.getName(), width / 2, 2, 0xFFFFFF);
		}
		drawString(fontRenderer, ""+record.player.getPosition()+
				" | "+record.player.chunkCoordX+", "+record.player.chunkCoordY+" "+record.player.chunkCoordZ, 2, 2, 0xFFFFFF);
		drawString(fontRenderer, record.from+" | "+record.to, 2, 15, 0xFFFFFF);
		drawString(fontRenderer, record.world.getDebugLoadedEntities(), 2, 28, 0xFFFFFF);
        itemRender.renderItemAndEffectIntoGUI(new ItemStack(Items.CLOCK), width-18, height-18);
		mc.world = null;
		mc.player = null;
		mc.playerController = null;
	}
	
	private void drawDebugVision() {
		int res = 24;
		float fov = 70;
		float ratio = (float) width / height;
		SensorRaytrace sensor = new SensorRaytrace(res, (int) (res / ratio), 1, fov, ratio, false) {
			
			@Override
			protected void encode(World world, Vec3d from, Vec3d dir, RayTraceResult res, float[] result) {
				if(res == null) {
					result[0] = -1;
					return;
				}
				IBlockState state = world.getBlockState(res.getBlockPos());
				if(state.getBlock() == Blocks.STAINED_GLASS) {
					result[0] = state.getValue(BlockColored.COLOR) == EnumDyeColor.YELLOW ? 1 : 0;
				} else {
					result[0] = -1;
				}
			}
		};
//	long timeStart = System.currentTimeMillis();
		sensor.sense(record.world, record.player.getPositionEyes(1), record.player.rotationYaw, record.player.rotationPitch, followedEntity == null ? record.player : followedEntity);
//	long elapsed = System.currentTimeMillis() - timeStart;
//	System.out.println(elapsed+" ms for "+depthSensor.dim+" raytrace ("+(elapsed * 1000 / depthSensor.dim)+" us per ray)");
//	System.out.println("max dist is: "+(depthSensor.d*depthSensor.hRes));
		int channel = 0;
		if(Keyboard.isKeyDown(Keyboard.KEY_1)) channel += 1;
		if(Keyboard.isKeyDown(Keyboard.KEY_2)) channel += 2;
		if(Keyboard.isKeyDown(Keyboard.KEY_3)) channel += 3;
		if(Keyboard.isKeyDown(Keyboard.KEY_4)) channel += 4;
		if(Keyboard.isKeyDown(Keyboard.KEY_5)) channel += 5;
		channel = channel % sensor.channels;
		float rW = width / (float) sensor.hRes;
		float rH = height / (float) sensor.vRes;
		for(int v = 0; v < sensor.vRes; v++) {
			for(int h = 0; h < sensor.hRes; h++) {
				float val = sensor.values[(v * sensor.hRes + h) * sensor.channels + channel];
				int r = (0x7F + Math.round(0x80 * Math.max(0, +val)));
				int g = 0x7F;
				int b = (0x7F + Math.round(0x80 * Math.max(0, -val)));
				int color = (r << 16) | (g << 8) | b;
				color |= 0xFF000000;
				drawRect((int) (h * rW), (int) (v * rH), (int) ((h+1) * rW), (int) ((v+1) * rH), color);
			}
		}
	}
	
	@Override
	public boolean doesGuiPauseGame() {
		return false;
	}
}